from typing import TypeVar, Type
from enum import Enum


def scrub(text: str) -> str:
    """Removes '{' and '}' from the text, which cause string format errors."""
    return text.replace("{", "").replace("}", "")


E = TypeVar("E", bound=Enum)


def enum_from_str(enum_type: Type[E], s: str) -> E:
    try:
        return enum_type[s.upper()]
    except KeyError:
        raise ValueError(f"Unsupported {enum_type}: {s}")
