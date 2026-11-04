import hashlib
import pytest
from docx import Document

from production_rag.ingestion import (
    hash_file_bytes,
    validate_pdf_signature,
    validate_docx_signature,
    validate_text_contents,
    validate_upload_content,
)


def test_hash_file_bytes_uses_exact_original_bytes(tmp_path):
    path = tmp_path / "sample.txt"
    original_bytes = b"hello\r\nworld\xff"
    path.write_bytes(original_bytes)

    expected_hash = hashlib.sha256(original_bytes).hexdigest()

    assert hash_file_bytes(path) == expected_hash


def test_pdf_signature_accepts_pdf_header(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.7")

    validate_pdf_signature(path)


def test_pdf_signature_rejects_plain_text_named_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"This is plain text")

    with pytest.raises(ValueError, match="PDF extension"):
        validate_pdf_signature(path)


def test_docx_signature_accepts_word_document(tmp_path):
    path = tmp_path / "sample.docx"
    document = Document()
    document.add_paragraph("sample")
    document.save(path)

    validate_docx_signature(path)


def test_docx_signature_rejects_plain_text_named_docx(tmp_path):
    path = tmp_path / "sample.docx"
    path.write_text("plain_text", encoding="utf-8")

    with pytest.raises(ValueError, match="DOCX extension"):
        validate_docx_signature(path)


def test_text_content_accepts_txt_and_csv(tmp_path):
    for extension in (".txt", ".csv"):
        path = tmp_path / f"sample{extension}"
        path.write_text("name, value\nSam,1\n", encoding="utf-8")

        validate_text_contents(path)


def test_text_content_rejects_binary_bytes(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_bytes(b"\xff\x00")

    with pytest.raises(ValueError, match="UTF-8"):
        validate_text_contents(path)


def test_validate_upload_content_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "sample.exe"
    path.write_bytes(b"some content")

    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_upload_content(path)
