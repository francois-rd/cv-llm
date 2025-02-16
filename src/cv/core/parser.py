from typing import Optional
import json
import re


class ScoreOutputParser:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[float]:
        """
        Parses the generated_text of an LLM to extract a numerical score.
        This score can represent a binary indicator (0 vs 1), a Likert Scale
        with a given range, or an unbounded quantity, for example.
        Returns None on parsing failure.
        """
        raise NotImplementedError


class PatternMatchParser(ScoreOutputParser):
    def __init__(self, pattern: str, score_group: int = 1, flags=re.IGNORECASE):
        """
        Uses a regex pattern to parse LLM output. Returns a score or None of failure.

        :param pattern: A regex pattern from which to extract a match.
        :param score_group: The group index of the score within the pattern Match object.
        :param flags: Flags to pass to re.search(), if any.
        """
        super().__init__()
        self.pattern = pattern
        self.score_group = score_group
        self.flags = flags

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[float]:
        if self.flags is None:
            match = re.search(self.pattern, generated_text)
        else:
            match = re.search(self.pattern, generated_text, flags=self.flags)
        try:
            return float(match.group(self.score_group))
        except (AttributeError, ValueError):
            return None


class ExactMatchParser(ScoreOutputParser):
    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[float]:
        """
        Returns the exactly matching number in the generated_text (barring whitespace)
        as a score or None if the text is not an exact match for a number.
        """
        try:
            return float(generated_text.strip())
        except ValueError:
            return None


class JSONParser(ScoreOutputParser):
    def __init__(
        self,
        schema_key: str,
        pattern: str = r"({.*?})",  # NOTE: Doesn't catch JSON objects w/ nested dicts.
        flags=re.IGNORECASE,
    ):
        """
        Extracts JSON objects from generated_text, checking whether the value at the
        schema_key in each object corresponds to a score. Returns None on failure.

        :param schema_key: The key into the JSON object containing the score.
        :param pattern: A regex pattern to extract JSON objects from generated_text
            that may also include other text.
        :param flags: Flags to pass to re.compile(), if any.
        """
        super().__init__()
        self.schema_key = schema_key
        if flags is None:
            self.pattern = re.compile(pattern)
        else:
            self.pattern = re.compile(pattern, flags=flags)

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[float]:
        for string in [generated_text, *self.pattern.findall(generated_text)]:
            try:
                return float(json.loads(string, **kwargs)[self.schema_key])
            except (AttributeError, KeyError, TypeError, ValueError, json.decoder.JSONDecodeError):
                continue
        return None


class DefaultScoreParser(ScoreOutputParser):
    # NOTE: These are in approximate descending order of confidence in the pattern.
    default_sub_parsers: list[ScoreOutputParser] = [
        ExactMatchParser(),
        JSONParser("score"),
        PatternMatchParser(r'"?score"?\s*:\s*"?(\w+)"?'),
        PatternMatchParser(r'Score:\s*"?(\w+)"?'),
        PatternMatchParser(r'Answer:\s*"?(\w+)"?'),
        PatternMatchParser(r'{\s*"?score"?\s*:\s*"?(\w+)"?\s*}'),
        PatternMatchParser(r'{\s*"?score"?\s*:\s*"(\w+)"\s*}?'),
        PatternMatchParser(r'score is:?\s*"?(\w+)"?'),
        PatternMatchParser(r'^\s*"?(\w+)"?\n'),
    ]

    def __init__(
        self,
        min_score: float,
        max_score: float,
        force_int: bool = False,
        int_tol: float = 0.0001,
        sub_parsers: list[ScoreOutputParser] = None,
    ):
        """
        Parses LLM output using sub-parsers. These are checked in order, and so should
        be given in descending order of confidence in their ability to extract a valid
        label from the generated text. Uses SimpleScoreParser.default_sub_parsers
        if sub_parsers is None.

        :param min_score: Minimum allowed score. Scores less than this are rejected.
        :param max_score: Maximum allowed score. Scores greater than this are rejected.
        :param force_int: Whether to fail if the score cannot be coerced to an integer.
        :param int_tol: Tolerance (away from a whole number) for a score to be
                        considered a valid integer.
        :param sub_parsers: Optional list of sub-parsers to call, in order.
        """
        super().__init__()
        self.min_score, self.max_score = min_score, max_score,
        self.force_int, self.int_tol = force_int, int_tol
        self.parsers = self.default_sub_parsers if sub_parsers is None else sub_parsers

    def __call__(self, generated_text: str, *args, **kwargs) -> Optional[float]:
        """Returns the score of the first successful sub-parser, or None on failure of all parsers."""
        for parser in self.parsers:
            score = parser(generated_text, *args, **kwargs)
            if score is not None and self.min_score <= score <= self.max_score:
                if self.force_int:
                    if abs(int(score) - score) < self.int_tol:
                        return score
                else:
                    return score
        return None
