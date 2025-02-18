from typing import TypeVar, Type


def scrub(text: str) -> str:
    """Removes '{' and '}' characters from the text, which cause langchain formatting errors."""
    return text.replace("{", "").replace("}", "")


E = TypeVar("E")


def enum_from_str(enum_type: Type[E], s: str) -> E:
    try:
        return enum_type[s.upper()]
    except KeyError:
        raise ValueError(f"Unsupported {enum_type}: {s}")
