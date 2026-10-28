from pathlib import Path
import logging

import pandas as pd
from pypdf import PdfReader

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class LoadedBlock:
    text: str
    location: str
    block_type: str = "text"


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def load_text_file(path: Path) -> list[LoadedBlock]:
    text = clean_text(path.read_text(encoding="utf-8"))
    if not text:
        raise ValueError(f"Text file is empty or only whitespaces: {path}")
    return [LoadedBlock(text=text, location="whole file")]


def load_csv_file(path: Path) -> list[LoadedBlock]:
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
