from pathlib import Path
import logging
import hashlib

from pypdf import PdfReader

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from dataclasses import dataclass
from zipfile import BadZipFile, ZipFile


logger = logging.getLogger(__name__)


@dataclass
class LoadedBlock:
    text: str
    location: str
    block_type: str = "text"


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def validate_pdf_signature(path: Path) -> None:
    with path.open("rb") as file:
        header = file.read(5)

    if header != b"%PDF-":
        raise ValueError("The file contents do not match the PDF extension")


def validate_docx_signature(path: Path) -> None:
    try:
        with ZipFile(path) as word_file:
            parts = word_file.namelist()
    except BadZipFile as error:
        raise ValueError("The file contents do not match the DOCX extension") from error

    if "[Content_Types].xml" not in parts or "word/document.xml" not in parts:
        raise ValueError("The file contents do not match the DOCX extension")


def validate_text_contents(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Text file is not valid UTF-8") from error

    if "\x00" in content:
        raise ValueError("Text file contains binary data")


def validate_upload_content(path: Path) -> None:
    extension = path.suffix.casefold()

    if extension == ".pdf":
        validate_pdf_signature(path)
    elif extension == ".docx":
        validate_docx_signature(path)
    elif extension in {".txt", ".csv"}:
        validate_text_contents(path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")


def hash_file_bytes(path: Path) -> str:
    file_hash = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            data = file.read(1024 * 1024)
            if not data:
                break
            file_hash.update(data)

    return file_hash.hexdigest()


def load_text_file(path: Path) -> list[LoadedBlock]:
    text = clean_text(path.read_text(encoding="utf-8"))
    if not text:
        raise ValueError(f"Text file is empty or only whitespaces: {path}")
    return [LoadedBlock(text=text, location="whole file")]


def load_csv_file(path: Path) -> list[LoadedBlock]:
    import pandas as pd

    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )
    blocks = []

    for row_number, record in enumerate(frame.to_dict(orient="records"), start=2):
        row_text = clean_text(
            " | ".join(f"{column}: {value}" for column, value in record.items())
        )
        blocks.append(
            LoadedBlock(
                text=row_text, location=f"row {row_number}", block_type="csv_row"
            )
        )

    if not blocks:
        raise ValueError(f"No data rows found in CSV: {path}")
    return blocks


def load_pdf_file(path: Path) -> list[LoadedBlock]:
    reader = PdfReader(str(path))
    blocks = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")
        if page_text:
            blocks.append(
                LoadedBlock(
                    text=page_text, location=f"page {page_number}", block_type="page"
                )
            )
    if not blocks:
        raise ValueError(f"No readable text found in PDF: {path}")
    return blocks


def load_word_file(path: Path) -> list(LoadedBlock):
    document = Document(path)
    blocks = []
    paragraph_number = 0
    table_number = 0

    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            paragraph_number += 1
            paragraph_text = clean_text(item.text)
            if paragraph_text:
                style_id = item.style.style_id
                block_type = (
                    "heading" if style_id.startswith("Heading") else "paragraph"
                )
                blocks.append(
                    LoadedBlock(
                        text=paragraph_text,
                        location=f"paragraph {paragraph_number}",
                        block_type=block_type,
                    )
                )
        elif isinstance(item, Table):
            table_number += 1
            table_rows = []

            for row in item.rows:
                cells = [clean_text(cell.text) for cell in row.cells]
                if any(cells):
                    table_rows.append(" | ".join(cells))
            if table_rows:
                blocks.append(
                    LoadedBlock(
                        text="\n".join(table_rows),
                        location=f"table {table_number}",
                        block_type="table",
                    )
                )

    if not blocks:
        raise ValueError(f"No readable text found in Word document: {path}")
    return blocks


def load_document(path: Path) -> list(LoadedBlock):

    try:
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return load_text_file(path)
        elif suffix == ".md":
            return load_text_file(path)
        elif suffix == ".csv":
            return load_csv_file(path)
        elif suffix == ".pdf":
            return load_pdf_file(path)
        elif suffix == ".docx":
            return load_word_file(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    except Exception:
        logger.exception("Skipping document: %s", path.name)
        return []
