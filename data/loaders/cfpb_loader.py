"""Load and clean CFPB consumer complaint CSV datasets."""

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"complaint_text", "category"}
_CFPB_COLUMN_MAP = {
    "Consumer complaint narrative": "complaint_text",
    "Product": "category",
}


def load_cfpb(path: str | Path) -> pd.DataFrame:
    """Load and clean a CFPB complaint CSV file.

    Handles both the raw CFPB column names (e.g. 'Consumer complaint narrative')
    and already-normalised column names. Rows with empty complaint text are dropped.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with at minimum columns: ``complaint_text``, ``category``, ``source``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns cannot be found after column mapping.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CFPB dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    logger.debug("Loaded %d rows from %s", len(df), path)

    # Rename raw CFPB column names if present
    df = df.rename(columns=_CFPB_COLUMN_MAP)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}. Found: {list(df.columns)}")

    df = df[list(_REQUIRED_COLUMNS)].copy()
    df["source"] = path.stem

    before = len(df)
    df = df.dropna(subset=["complaint_text"])
    df = df[df["complaint_text"].str.strip() != ""]
    logger.debug("Dropped %d rows with empty complaint text", before - len(df))

    df = df.reset_index(drop=True)
    logger.info("Loaded %d complaints from %s", len(df), path.name)
    return df


def load_and_combine(paths: list[str | Path]) -> pd.DataFrame:
    """Load and concatenate multiple CFPB CSV files.

    Parameters
    ----------
    paths : list of str or Path
        Paths to CSV files.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame deduplicated by complaint_text.
    """
    frames = [load_cfpb(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["complaint_text"]).reset_index(drop=True)
    logger.info("Combined %d complaints (%d duplicates dropped)", len(combined), before - len(combined))
    return combined
