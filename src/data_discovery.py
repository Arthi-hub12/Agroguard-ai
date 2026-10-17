"""Dataset discovery: recursively finds supported data files and loads them safely.

Nothing here assumes filenames, columns or paths. If no files are found the
result says so and explains what to do; it never invents data.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from config import settings


@dataclass
class DiscoveryResult:
    searched_dir: str
    working_dir: str
    exists: bool
    files: list = field(default_factory=list)          # list[dict]
    inaccessible: list = field(default_factory=list)   # list[dict]
    counts_by_type: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)

    @property
    def found_any(self) -> bool:
        return len(self.files) > 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def discover_datasets(raw_dir: Optional[Path | str] = None,
                      extensions: Optional[dict] = None) -> DiscoveryResult:
    """Recursively search ``raw_dir`` for supported dataset files."""
    extensions = extensions or settings.SUPPORTED_EXTENSIONS
    raw_path = Path(raw_dir).expanduser() if raw_dir else Path(settings.RAW_DIR)
    result = DiscoveryResult(
        searched_dir=str(raw_path.resolve() if raw_path.exists() else raw_path),
        working_dir=os.getcwd(),
        exists=raw_path.exists(),
    )

    if not raw_path.exists():
        result.messages.append(
            f"The folder '{raw_path}' does not exist. Create it, or point AgroGuard to the "
            "right folder (see 'How to set the data folder' below)."
        )
        result.messages.append(_how_to_configure())
        return result
    if not raw_path.is_dir():
        result.exists = False
        result.messages.append(f"'{raw_path}' is a file, not a folder. Provide a folder path.")
        return result

    def _on_error(err: OSError) -> None:
        result.inaccessible.append({"path": str(getattr(err, "filename", "?")), "error": str(err)})

    for root, _dirs, filenames in os.walk(raw_path, onerror=_on_error):
        for name in sorted(filenames):
            p = Path(root) / name
            ext = p.suffix.lower()
            if ext not in extensions:
                continue
            if not os.access(p, os.R_OK):
                result.inaccessible.append({"path": str(p), "error": "no read permission"})
                continue
            try:
                size = p.stat().st_size
            except OSError as exc:
                result.inaccessible.append({"path": str(p), "error": str(exc)})
                continue
            result.files.append({
                "name": name,
                "path": str(p),
                "relative_path": str(p.relative_to(raw_path)),
                "type": extensions[ext],
                "extension": ext,
                "size_bytes": size,
            })

    for f in result.files:
        result.counts_by_type[f["type"]] = result.counts_by_type.get(f["type"], 0) + 1

    if not result.files:
        result.messages.append(
            f"No supported files ({', '.join(sorted(extensions))}) were found under '{raw_path}'."
        )
        result.messages.append(
            "Likely causes: (1) the dataset has not been copied into the folder yet, "
            "(2) the folder path is wrong, (3) the files use an unsupported extension "
            "(for example .zip - unzip it first)."
        )
        result.messages.append(_how_to_configure())
    return result


def _how_to_configure() -> str:
    return (
        "How to set the data folder: copy your dataset files into "
        f"'{settings.PROJECT_ROOT / 'data' / 'raw'}', OR set the environment variable "
        "AGROGUARD_RAW_DIR to your folder (PowerShell: $env:AGROGUARD_RAW_DIR='D:\\\\datasets'; "
        "Linux: export AGROGUARD_RAW_DIR=/path/to/datasets), OR pass --raw-dir to the CLI."
    )


def print_discovery_summary(result: DiscoveryResult) -> None:
    print("=" * 70)
    print("AgroGuard dataset discovery")
    print(f"  Working directory : {result.working_dir}")
    print(f"  Searched folder   : {result.searched_dir}")
    print(f"  Folder exists     : {result.exists}")
    print(f"  Supported files   : {len(result.files)}  {result.counts_by_type}")
    if result.inaccessible:
        print(f"  Inaccessible      : {len(result.inaccessible)}")
        for item in result.inaccessible:
            print(f"    - {item['path']}: {item['error']}")
    for f in result.files:
        print(f"    * {f['relative_path']} ({f['type']}, {f['size_bytes']:,} bytes)")
    for msg in result.messages:
        print(f"  NOTE: {msg}")
    print("=" * 70)


def load_dataset(path: Path | str) -> tuple[Optional[pd.DataFrame], list]:
    """Load one file. Returns (DataFrame or None, list of warning/error strings).

    Never raises for bad files: problems are reported as strings.
    """
    p = Path(path)
    warnings: list = []
    ext = p.suffix.lower()
    try:
        if p.stat().st_size == 0:
            return None, ["File is empty (0 bytes)."]
    except OSError as exc:
        return None, [f"Cannot access file: {exc}"]

    try:
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else None
            df = _read_csv_flexible(p, sep, warnings)
        elif ext in (".xlsx", ".xls"):
            try:
                df = pd.read_excel(p)
            except ImportError as exc:
                return None, [f"Missing optional dependency for Excel files: {exc}. "
                              "Install with: pip install openpyxl"]
        elif ext == ".json":
            df = _read_json_flexible(p, warnings)
        elif ext == ".parquet":
            try:
                df = pd.read_parquet(p)
            except ImportError as exc:
                return None, [f"Parquet support needs pyarrow: {exc}. Install with: pip install pyarrow"]
        else:
            return None, [f"Unsupported file type: {ext}"]
    except Exception as exc:  # noqa: BLE001 - we want to report, not crash
        return None, [f"Could not parse file: {type(exc).__name__}: {exc}"]

    if df is None or df.shape[1] == 0:
        return None, warnings + ["File parsed but contains no columns."]
    if len(df) == 0:
        warnings.append("File has headers but zero data rows.")
    return df, warnings


def _read_csv_flexible(p: Path, sep: Optional[str], warnings: list) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(p, sep=sep, engine="python", encoding=encoding,
                             on_bad_lines="skip")
            if encoding != "utf-8":
                warnings.append(f"File was read using '{encoding}' encoding.")
            return df
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
    raise last_exc or ValueError("Could not decode file")


def _read_json_flexible(p: Path, warnings: list) -> pd.DataFrame:
    try:
        return pd.read_json(p)
    except ValueError:
        pass
    try:
        return pd.read_json(p, lines=True)
    except ValueError:
        pass
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                warnings.append("JSON records were found inside a nested key.")
                return pd.DataFrame(value)
        return pd.json_normalize(data)
    return pd.json_normalize(data)
