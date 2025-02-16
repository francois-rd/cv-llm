from typing import Iterable, Optional
from dataclasses import dataclass
import re

from omegaconf import OmegaConf

from .base import TagsConfig, Transcript
from ..core import Cluster, ClusterName, ClustersConfig, QuestionId


@dataclass
class Tag:
    question_ids: Optional[list[QuestionId]]
    match_string: str


Span = tuple[Optional[int], Optional[int]]


class Tagger:
    def __init__(self, tags_cfg: TagsConfig):
        self.tags_cfg = tags_cfg
        self.pattern = re.compile(tags_cfg.primary_regex, flags=re.IGNORECASE)
        self.digits = re.compile(tags_cfg.question_id_regex, flags=re.IGNORECASE)

    def __call__(self, lines: list[str], *args, **kwargs) -> list[Optional[Tag]]:
        return [self._do_tag(line) for line in lines]

    def _do_tag(self, line) -> Optional[Tag]:
        match = self.pattern.search(line)
        if match is None:
            return None
        question_tag = match.group(self.tags_cfg.question_group)
        try:
            return Tag([int(question_tag)], match.group())
        except ValueError:
            res = self.digits.findall(question_tag)
            if len(set(res)) == 1:  # All elements equal and there is at least one.
                return Tag([int(res[0])], match.group())
            return Tag([int(r) for r in res], match.group())


class ConvertTagsToTranscript:
    """
    Several big assumptions:
    1. Lines within a single question have a consecutive span, as opposed to being
       fragmented across the transcript.
    2. Clusters can have non-consecutive questions (e.g., Q1, Q6, Q7), but the
       resulting 'cluster.lines' will contain lines in ascending order regardless of
       the listed order of 'cluster.questions'.
    3. If a tag identifies more than one question for a single line of transcript,
       then that line is repeated across each cluster to which the question belongs.
       In other words, mix-tag lines don't have to belong to the same cluster.
       Instead, their content is replicated across clusters.
    """
    def __init__(self, clusters_cfg: ClustersConfig):
        self.clusters_cfg = clusters_cfg
        self.q_id_to_cluster_map = {}
        for name, data in clusters_cfg.clusters.items():
            for question_id in data.questions:
                self.q_id_to_cluster_map[question_id] = name

    def __call__(
        self,
        lines: list[str],
        tags: list[Optional[Tag]],
        *args,
        **kwargs,
    ) -> Transcript:
        transcript = {}
        for name, data in self.clusters_cfg.clusters.items():
            transcript[name] = Cluster(OmegaConf.to_object(data))
        lines = [self._remove_tag(line, i, tags) for i, line in enumerate(lines)]
        self._fill_transcript(transcript, lines, tags)
        return Transcript(transcript)

    @staticmethod
    def _remove_tag(line: str, index, tags: list[Optional[Tag]]) -> str:
        tag = tags[index]
        if tag is None:
            return line
        else:
            return line.replace(tag.match_string, "").strip()

    def _fill_transcript(
        self,
        transcript: dict[ClusterName, Cluster],
        lines: list[str],
        tags: list[Optional[Tag]],
    ):
        all_spans = self._find_all_spans(tags)
        for cluster in transcript.values():
            self._fill_cluster(cluster, lines, all_spans)

    def _find_all_spans(self, tags: list[Optional[Tag]]):
        return {
            q_id: self._find_span(q_id, tags)
            for q_id in self.q_id_to_cluster_map
        }

    @staticmethod
    def _find_span(q_id: QuestionId, tags: list[Optional[Tag]]) -> Span:
        start, end = None, None
        for i, tag in enumerate(tags):
            if tag is None:  # If we have a blank tag...
                # ... increment the end tag only if we've found the start already.
                end = None if start is None else i
                continue  # Importantly, go to next loop.

            if q_id not in tag.question_ids:  # Tag isn't blank, but q_ids not in it...
                if start is None and end is None:
                    continue  # ... continue if we haven't hit the span at all yet...
                break  # ... otherwise, we are done searching.

            # If q_id IS IN tag, then we set the start (if it's the first time we
            # hit a tag with q_id), or we increment the end (if start has been set).
            if start is None:
                start = i
            else:
                end = i
        return start, end

    def _fill_cluster(
        self,
        cluster: Cluster,
        lines: list[str],
        all_spans: dict[QuestionId, Span],
    ):
        spans = self._find_cluster_spans(cluster, all_spans)
        for span in self._merge_and_sort_overlapping_spans(spans):
            cluster.lines.extend(lines[span[0]:span[1] + 1])

    @staticmethod
    def _find_cluster_spans(
        cluster: Cluster,
        all_spans: dict[QuestionId, Span],
    ) -> list[Span]:
        spans = []
        for q_id in cluster.data.questions:
            start, end = all_spans[q_id]
            if start is not None and end is not None:
                spans.append((start, end))
        return spans

    @staticmethod
    def _merge_and_sort_overlapping_spans(spans: list[Span]) -> Iterable[Span]:
        """
        Merges overlapping (and adjacent) spans. Yields the merged spans *in order*.
        """
        # NOTE: Implementation borrows from:
        #  https://codereview.stackexchange.com/questions/21307/consolidate-list-of-ranges-that-overlap
        spans = iter(sorted(spans))
        try:
            # The 'try' fails if 'spans' starts off empty.
            merged_start, merged_end = next(spans)
        except StopIteration:
            # Since this is a generator, return.
            # In Py>=3.7, do NOT raise StopIteration.
            return

        # NOTE: The below works *only* because the spans are sorted.
        for start, end in spans:
            # If the new start is larger than the current end, there is a gap.
            if start > merged_end:
                # Yield the current merged span and start a new one.
                yield merged_start, merged_end
                merged_start, merged_end = start, end
            else:
                # Otherwise, the spans are either adjacent or fully overlapping. Merge.
                merged_end = max(merged_end, end)
        # As we fall off the end of the loop, yield the very last merged span.
        yield merged_start, merged_end
