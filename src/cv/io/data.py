from typing import Any, Callable, Type, TypeVar
from dataclasses import asdict, dataclass
from pathlib import Path
from enum import Enum
import json
import os

from docx import Document
from dacite import Config, from_dict
import pandas as pd


@dataclass
class PathConfig:
    """Programmatic access to file paths of important files. Set `root_dir' appropriately."""

    # Root directory for all data.
    root_dir: str = ""

    # Resources (databases and datasets).
    resources_dir: str = "${root_dir}/resources"

    # Folder containing the transcript data.
    transcript_dir: str = "${resources_dir}/transcripts"

    # Folder containing the original transcript data.
    docx_transcript_dir: str = "${transcript_dir}/docx"

    # Folder containing the JSON-converted transcript data.
    json_transcript_dir: str = "${transcript_dir}/json"

    # Folder containing the clustered transcript data.
    clustered_transcript_dir: str = "${transcript_dir}/clustered"

    # Results (from LLMs and evaluation).
    results_dir: str = "${root_dir}/results"

    # Top-level folder for storing the raw LLM output. Each sub-folder is the
    # Nickname of a particular LLM.
    raw_scores_dir: str = "${results_dir}/raw_scores/${run_id}"

    # The nickname or ID of the current run. Update this at runtime on the command line.
    run_id: str = "<missing-run-id>"


T = TypeVar("T")

# Default dacite config adds support for both Enum and tuple types.
DEFAULT_CONFIG = Config(cast=[Enum, tuple])


def enum_dict_factory(data) -> dict:
    """
    Recursively checks for Enums and converts any that are found to their respective
    values. NOTE: Recursion operates only on objects of type Dict and List.
    """
    new_data = []
    for k, v in data:
        if isinstance(v, Enum):
            new_data.append((k, v.value))
        elif isinstance(v, dict):
            new_data.append((k, enum_dict_factory(list(v.items()))))
        elif isinstance(v, list):
            result = enum_dict_factory([(kk, vv) for kk, vv in enumerate(v)])
            new_data.append((k, [result[i] for i in range(len(v))]))
        else:
            new_data.append((k, v))
    return dict(new_data)


def ensure_path(file_path: str) -> str:
    """Creates all parent folders of a file path (if needed). Returns the file path."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    return file_path


def save_lines(file_path: str, *objs: T, to_string_fn: Callable[[T], str] = str):
    """Saves each object as a string to the file path, one object per line."""
    str_objs = [to_string_fn(o) + os.linesep for o in objs]
    with open(ensure_path(file_path), "w", encoding="utf-8") as f:
        f.writelines(str_objs)


def save_json(file_path: str, obj, **kwargs):
    """Saves the object to the file path. kwargs are passed to json.dump()."""
    with open(ensure_path(file_path), "w", encoding="utf-8") as f:
        json.dump(obj, f, **kwargs)


def save_jsonl(file_path: str, *objs: Any, **kwargs):
    """
    Saves each object as a JSON string to the file path, one object per line.
    kwargs are passed to json.dumps().
    """
    save_lines(file_path, *objs, to_string_fn=lambda o: json.dumps(o, **kwargs))


def save_dataclass_json(
    file_path: str,
    obj: Any,
    dict_factory: Callable = enum_dict_factory,
    **kwargs,
):
    """
    Saves the dataclass object to the file path. kwargs are passed to json.dumps().
    dict_factory is passed to dataclasses.asdict().
    """
    with open(ensure_path(file_path), "w", encoding="utf-8") as f:
        json.dump(asdict(obj, dict_factory=dict_factory), f, **kwargs)


def save_dataclass_jsonl(
    file_path: str,
    *objs: Any,
    dict_factory: Callable = enum_dict_factory,
    **kwargs,
):
    """
    Saves each dataclass object as a JSON string to the file path, one object per line.
    kwargs are passed to json.dump(). dict_factory is passed to dataclasses.asdict().
    """

    def helper(o):
        return json.dumps(asdict(o, dict_factory=dict_factory), **kwargs)

    save_lines(file_path, *objs, to_string_fn=helper)


def dumps_dataclasses(*objs: Any, dict_factory: Callable = enum_dict_factory, **kwargs) -> str:
    """
    Converts *objs to a string that represents a JSON list of dicts, one for each object.
    kwargs are passed to json.dump(). dict_factory is passed to dataclasses.asdict().
    """

    return json.dumps([asdict(o, dict_factory=dict_factory) for o in objs], **kwargs)


def to_dicts(*objs: Any, dict_factory: Callable = enum_dict_factory) -> list[dict[str, Any]]:
    """Converts each dataclass object to a plain dict. dict_factory is passed to dataclasses.asdict()"""
    return [asdict(o, dict_factory=dict_factory) for o in objs]


def load_lines(file_path: str) -> list[str]:
    """Loads each line from the file path, one string per line."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()


def load_json(file_path: str, **kwargs) -> Any:
    """Loads a JSON object from the file path. kwargs are passed to json.load()."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f, **kwargs)


def load_jsonl(file_path: str, **kwargs) -> list[Any]:
    """Loads JSON objects from the file path. kwargs are passed to json.loads()."""
    return [json.loads(line.strip(), **kwargs) for line in load_lines(file_path)]


def load_dataclass_json(
    file_path: str,
    t: Type[T],
    dacite_config: Config = DEFAULT_CONFIG,
    **kwargs,
) -> T:
    """
    Loads a dataclass object of type 't' from the file path. kwargs are passed
    to json.load(). dacite_config is passed to dacite.from_dict().
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return from_dict(t, json.load(f, **kwargs), config=dacite_config)


def load_dataclass_jsonl(
    file_path: str,
    t: Type[T],
    dacite_config: Config = DEFAULT_CONFIG,
    **kwargs,
) -> list[T]:
    """
    Loads dataclass objects of type 't' from the file path. kwargs are passed
    to json.loads(). dacite_config is passed to dacite.from_dict().
    """
    return [from_dict(t, json.loads(line.strip(), **kwargs), config=dacite_config) for line in load_lines(file_path)]


def loads_dataclass_jsonl(
    s: str,
    t: Type[T],
    dacite_config: Config = DEFAULT_CONFIG,
    **kwargs,
) -> list[T]:
    """
    Loads dataclass objects of type 't' from 's', assuming 's' represents as valid list of dicts of
    objects of type 't'. kwargs are passed to json.loads(). dacite_config is passed to dacite.from_dict().
    """
    return [from_dict(t, d, config=dacite_config) for d in json.loads(s, **kwargs)]


def from_dicts(t: Type[T], *dicts: dict[str, Any], dacite_config: Config = DEFAULT_CONFIG) -> list[T]:
    """
    Converts each dict in 'dicts' into a dataclass object of type 't'. dacite_config is passed to dacite.from_dict().
    """
    # noinspection PyTypeChecker
    # dacite.from_dict expects a Protocol type (dacite.data.Data) which just mimics the dictionary interface.
    return [from_dict(t, d, config=dacite_config) for d in dicts]


def load_records_csv(file_path: str, **kwargs) -> list[dict]:
    """
    Loads records (one per CSV row) from the file path. kwargs are passed to
    pandas.read_csv().
    """
    return pd.read_csv(file_path, **kwargs).to_dict(orient="records")


def load_docx(file_path: str) -> list[str]:
    lines = [paragraph.text for paragraph in Document(file_path).paragraphs]
    return [line.strip() for line in lines if line.strip()]
