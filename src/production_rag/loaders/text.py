from pathlib import Path


def load_text_file(path: Path) -> str:
    """Read a UTF-8 text file and return its contents."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Text file is empty or only whitespaces: {path}")
    return text
