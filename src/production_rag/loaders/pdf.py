from pathlib import Path

from pypdf import PdfReader


def load_pdf_file(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append((page_number, page_text))
    return pages
