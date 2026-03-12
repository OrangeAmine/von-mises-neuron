"""Real-world dataset loading, caching, and preprocessing.

Downloads, caches, and prepares three UCI / public datasets for
benchmarking the VonMisesForecaster against baselines:

1. **UCI Occupancy Detection** -- indoor CO2 with daily occupancy cycles.
   ~20,000 one-minute rows -> ~432 hourly rows after resampling.
2. **UCI Household Electric Power Consumption** -- 7 electrical features
   at 1-min resolution -> ~35,000 hourly rows.  Daily + weekly periodicity.
3. **Jena Climate** -- temperature + pressure + humidity + wind at 10-min
   resolution -> ~70,000 hourly rows.  Diurnal + seasonal cycles.

Each loader returns a pandas DataFrame with a DatetimeIndex at hourly
resolution, ready for the standard pipeline (normalize -> create_sequences).

Downloads use ``urllib`` (stdlib) to avoid adding ``requests`` as a
dependency.  Files are cached in ``data/raw/<dataset>/`` and reused
on subsequent runs.
"""

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.preprocessing import resample_hourly


# ---------------------------------------------------------------------------
# Default data directory (relative to project root)
# ---------------------------------------------------------------------------
_DEFAULT_DATA_DIR = Path("data/raw")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download_file(url: str, dest: Path, timeout: int = 120) -> Path:
    """Download a file from *url* to *dest*, skipping if already cached.

    Args:
        url: Remote URL to download.
        dest: Local file path for the downloaded file.
        timeout: Socket timeout in seconds.

    Returns:
        The *dest* path.
    """
    if dest.exists():
        print(f"  Cache hit: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest} ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


def _extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract a zip archive to *dest_dir*, skipping if already extracted.

    Args:
        zip_path: Path to the .zip file.
        dest_dir: Directory to extract into.

    Returns:
        The *dest_dir* path.
    """
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"  Already extracted: {dest_dir}")
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {zip_path.name} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return dest_dir


# ---------------------------------------------------------------------------
# Loader: UCI Occupancy Detection
# ---------------------------------------------------------------------------
_OCCUPANCY_URL = (
    "https://archive.ics.uci.edu/static/public/357/"
    "occupancy+detection.zip"
)


def load_occupancy(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Download and preprocess the UCI Occupancy Detection dataset.

    The raw data has 3 CSV files (training, test1, test2) at 1-minute
    resolution.  We concatenate them, drop duplicates, remove the
    binary ``Occupancy`` column (it is a label, not a feature), and
    resample to hourly resolution.

    After resampling, ~432 hourly rows remain (~18 days).

    Args:
        data_dir: Root data directory (default ``data/raw``).

    Returns:
        DataFrame with hourly DatetimeIndex and columns
        [Temperature, Humidity, Light, CO2, HumidityRatio].
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    dataset_dir = data_dir / "occupancy"

    zip_path = _download_file(_OCCUPANCY_URL, dataset_dir / "occupancy.zip")
    extract_dir = _extract_zip(zip_path, dataset_dir / "extracted")

    # The zip contains a nested folder structure -- find all CSV files
    csv_files = sorted(extract_dir.rglob("*.txt")) + sorted(extract_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV/TXT files found in {extract_dir}. "
            "The archive structure may have changed."
        )

    frames = []
    for csv_path in csv_files:
        try:
            # Column 0 is a row number; "date" column has the timestamps
            df = pd.read_csv(csv_path)
            # Only keep files that look like occupancy data (have CO2 column)
            if "CO2" not in df.columns:
                continue

            # Find the datetime column (named "date" in this dataset)
            date_col = None
            for col in df.columns:
                if col.lower() == "date":
                    date_col = col
                    break
            if date_col is None:
                # Fallback: try parsing the first string-like column
                for col in df.columns:
                    if df[col].dtype == object:
                        date_col = col
                        break

            if date_col is not None:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                # Drop any unnamed row-number columns
                df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

            frames.append(df)
            print(f"  Loaded {csv_path.name}: {len(df)} rows")
        except Exception:
            continue  # skip non-data files (e.g. readme)

    if not frames:
        raise ValueError("No valid occupancy data files found in archive")

    df = pd.concat(frames, axis=0)
    df = df.sort_index()

    # Remove duplicate timestamps (overlap between training/test files)
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"  Removed {n_dupes} duplicate timestamps")

    # Drop the binary Occupancy column -- it's a classification label,
    # not a sensor reading, and would leak information about CO2 patterns
    if "Occupancy" in df.columns:
        df = df.drop(columns=["Occupancy"])

    # Ensure DatetimeIndex is timezone-naive for consistent resampling
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Resample 1-min -> 1h
    df = resample_hourly(df, interval="1h")

    # Ensure numeric types
    df = df.apply(pd.to_numeric, errors="coerce")

    print(f"  Final shape: {df.shape} "
          f"({len(df)} hours, columns={list(df.columns)})")
    return df


# ---------------------------------------------------------------------------
# Loader: UCI Household Electric Power Consumption
# ---------------------------------------------------------------------------
_POWER_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00235/household_power_consumption.zip"
)


def load_power_consumption(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Download and preprocess the UCI Household Electric Power dataset.

    The raw data is a single semicolon-delimited file at 1-minute
    resolution with ``?`` as the missing value marker.  We parse the
    Date + Time columns into a DatetimeIndex, convert ``?`` -> NaN,
    and resample to hourly.

    After resampling, ~35,000 hourly rows remain (~4 years).

    Args:
        data_dir: Root data directory (default ``data/raw``).

    Returns:
        DataFrame with hourly DatetimeIndex and 7 numeric columns.
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    dataset_dir = data_dir / "power"

    zip_path = _download_file(_POWER_URL, dataset_dir / "power.zip")
    extract_dir = _extract_zip(zip_path, dataset_dir / "extracted")

    # Find the main data file
    txt_files = sorted(extract_dir.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {extract_dir}. "
            "The archive structure may have changed."
        )

    data_file = txt_files[0]
    print(f"  Parsing {data_file.name}...")

    # Read with semicolon delimiter; '?' marks missing values
    df = pd.read_csv(
        data_file,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    # Combine Date + Time into a single DatetimeIndex
    # Format: DD/MM/YYYY HH:MM:SS
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    df = df.set_index("datetime")
    df = df.drop(columns=["Date", "Time"])

    # Convert all columns to numeric (some may still be strings)
    df = df.apply(pd.to_numeric, errors="coerce")

    print(f"  Raw data: {len(df)} rows, {df.isna().sum().sum()} total NaN values")

    # Resample 1-min -> 1h
    df = resample_hourly(df, interval="1h")

    print(f"  Final shape: {df.shape} "
          f"({len(df)} hours, columns={list(df.columns)})")
    return df


# ---------------------------------------------------------------------------
# Loader: Jena Climate
# ---------------------------------------------------------------------------
_JENA_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)

# The 5 most informative features for periodic forecasting:
#   T (degC)  -- target, strong diurnal + seasonal cycle
#   p (mbar)  -- atmospheric pressure (slow trends)
#   rh (%)    -- relative humidity (anti-correlated with T)
#   wv (m/s)  -- wind velocity
#   wd (deg)  -- wind direction (inherently circular/periodic)
_JENA_FEATURES = ["T (degC)", "p (mbar)", "rh (%)", "wv (m/s)", "wd (deg)"]


def load_jena_climate(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Download and preprocess the Jena Climate dataset.

    The raw data is a single CSV at 10-minute resolution (2009-2017).
    We select 5 key features, parse the datetime, and resample to
    hourly resolution.

    After resampling, ~70,000 hourly rows remain (~8 years).

    Args:
        data_dir: Root data directory (default ``data/raw``).

    Returns:
        DataFrame with hourly DatetimeIndex and 5 feature columns.
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    dataset_dir = data_dir / "jena"

    zip_path = _download_file(_JENA_URL, dataset_dir / "jena_climate.zip")
    extract_dir = _extract_zip(zip_path, dataset_dir / "extracted")

    # Find the CSV file
    csv_files = sorted(extract_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {extract_dir}. "
            "The archive structure may have changed."
        )

    data_file = csv_files[0]
    print(f"  Parsing {data_file.name}...")

    df = pd.read_csv(data_file)

    # Parse datetime -- format is DD.MM.YYYY HH:MM:SS
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time")

    # Select feature subset
    missing = [f for f in _JENA_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {missing} not found. "
            f"Available: {list(df.columns)}"
        )
    df = df[_JENA_FEATURES]

    # Handle known anomalies: wv (m/s) has some -9999.0 sentinel values
    df = df.replace(-9999.0, float("nan"))

    print(f"  Raw data: {len(df)} rows, {df.isna().sum().sum()} total NaN values")

    # Resample 10-min -> 1h
    df = resample_hourly(df, interval="1h")

    # Forward-fill any remaining NaN gaps (rare, <0.01% of data)
    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        print(f"  Forward-filling {n_nan} remaining NaN values")
        df = df.ffill().bfill()

    print(f"  Final shape: {df.shape} "
          f"({len(df)} hours, columns={list(df.columns)})")
    return df


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "occupancy": {
        "loader": load_occupancy,
        "target_col": "CO2",
        "unit": "ppm",
        # Reduced from 72/12 because only ~432 hourly rows after resampling.
        # With lookback=72 and 80/10/10 split, val/test would have 0 sequences.
        "lookback": 24,
        "horizon": 6,
        "handcrafted_periods": [24],          # daily cycle only
        "split_ratios": [0.7, 0.15, 0.15],   # wider val/test for small dataset
        "description": "UCI Occupancy Detection -- indoor CO2 with daily cycles",
    },
    "power": {
        "loader": load_power_consumption,
        "target_col": "Global_active_power",
        "unit": "kW",
        "lookback": 168,                      # 1 week -- captures weekly pattern
        "horizon": 24,                        # day-ahead forecast
        "handcrafted_periods": [24, 168],     # daily + weekly
        "split_ratios": [0.8, 0.1, 0.1],
        "description": "UCI Household Electric Power Consumption",
    },
    "jena": {
        "loader": load_jena_climate,
        "target_col": "T (degC)",
        "unit": "degC",
        "lookback": 168,                      # 1 week
        "horizon": 24,                        # day-ahead
        "handcrafted_periods": [24, 168, 8766],  # daily + weekly + annual
        "split_ratios": [0.8, 0.1, 0.1],
        "description": "Jena Climate 2009-2016 -- temperature forecasting",
    },
}


def load_dataset(
    name: str,
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a dataset by name from the registry.

    Args:
        name: Dataset identifier (one of "occupancy", "power", "jena").
        data_dir: Root data directory (default ``data/raw``).

    Returns:
        Tuple of (DataFrame, config_dict) where config_dict contains
        target_col, unit, lookback, horizon, split_ratios, etc.

    Raises:
        ValueError: If *name* is not in the registry.
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[name]
    loader: Callable = config["loader"]

    print(f"\n{'=' * 60}")
    print(f"  Loading dataset: {name}")
    print(f"  {config['description']}")
    print(f"{'=' * 60}")

    df = loader(data_dir=data_dir)

    # Return config without the loader function (not serializable)
    config_out = {k: v for k, v in config.items() if k != "loader"}
    return df, config_out


def list_datasets() -> List[str]:
    """Return a sorted list of available dataset names."""
    return sorted(DATASET_CONFIGS.keys())
