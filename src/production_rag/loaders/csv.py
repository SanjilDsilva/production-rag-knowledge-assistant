from pathlib import Path
import pandas as pd


def load_csv_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)
