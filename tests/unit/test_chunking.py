from production_rag.chunking import count_tokens, split_oversized_text
from production_rag.chunking import build_chunk_texts
from production_rag.ingestion import LoadedBlock, load_document
from production_rag.chunking import chunk_blocks
from production_rag.chunking import pack_text_pieces


class SimpleTokenizer:
    model_max_length = 512

    def encode(self, text, add_special_tokens=True):
        token_ids = list(range(len(text.split())))
        if add_special_tokens:
            token_ids.extend([-1, -2])
        return token_ids

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def decode(self, token_ids):
        return " ".join("word" for _ in token_ids)


def test_huge_text_without_punctuation_is_split_within_limit():
    tokenizer = SimpleTokenizer()
    text = "word " * 100

    pieces = split_oversized_text(text, tokenizer, max_tokens=20)

    assert len(pieces) > 1
    assert all(count_tokens(piece, tokenizer) <= 20 for piece in pieces)
    assert sum(len(piece.split()) for piece in pieces) == 100


def test_table_rows_stay_together_when_they_fit():
    tokenizer = SimpleTokenizer()
    block = LoadedBlock(
        text="Name | Role\nAda | Engineer",
        location="table 1",
        block_type="table",
    )

    chunks = build_chunk_texts([block], tokenizer, max_tokens=20, overlap_tokens=3)

    assert len(chunks) == 1
    text, location = chunks[0]
    assert "Name | Role" in text
    assert "Ada | Engineer" in text
    assert location == "table 1"


def test_empty_text_file_is_skipped(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")

    assert load_document(path) == []


def test_bad_encoding_text_file_is_skipped(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff")

    assert load_document(path) == []


def test_chunk_metadata_and_ids_are_stable():
    tokenizer = SimpleTokenizer()
    blocks = [
        LoadedBlock(
            text="Ada builds reliable software for her engineering team.",
            location="paragraph 1",
            block_type="paragraph",
        )
    ]

    first_run = chunk_blocks(
        blocks,
        document_id="doc-001",
        tenant_id="tenant-1",
        user_id="user-1",
        tokenizer=tokenizer,
    )
    second_run = chunk_blocks(
        blocks,
        document_id="doc-001",
        tenant_id="tenant-1",
        user_id="user-1",
        tokenizer=tokenizer,
    )

    chunk = first_run[0]
    assert chunk.chunk_id == second_run[0].chunk_id
    assert chunk.position == 0
    assert chunk.location == "paragraph 1"
    assert chunk.tenant_id == "tenant-1"
    assert chunk.user_id == "user-1"
    assert len(chunk.content_hash) == 64


def test_pieces_overlap_and_stay_within_token_limit():
    tokenizer = SimpleTokenizer()
    pieces = [
        ("one two three four five six seven eight nine", "paragraph 1"),
        ("alpha beta gamma delta epsilon zeta eta theta iota", "paragraph 2"),
        ("red green blue black white orange purple brown gray", "paragraph 3"),
    ]

    chunks = pack_text_pieces(pieces, tokenizer, max_tokens=20, overlap_tokens=3)

    assert len(chunks) == 2
    assert all(count_tokens(text, tokenizer) <= 20 for text, _ in chunks)
    assert chunks[1][0].startswith("word word word\n\n")


def test_csv_rows_remain_separate():
    tokenizer = SimpleTokenizer()
    rows = [
        LoadedBlock("name Ada role engineer", "row 2", "csv_row"),
        LoadedBlock("name Linus role developer", "row 3", "csv_row"),
    ]

    chunks = build_chunk_texts(rows, tokenizer, max_tokens=20, overlap_tokens=3)

    assert len(chunks) == 2
    assert [location for _, location in chunks] == ["row 2", "row 3"]


def test_same_text_in_different_tenants_gets_different_chunk_ids():
    tokenizer = SimpleTokenizer()
    blocks = [
        LoadedBlock(
            text="Ada builds reliable software for her engineering team.",
            location="paragraph 1",
            block_type="paragraph",
        )
    ]

    tenant_a_chunks = chunk_blocks(
        blocks,
        document_id="same-document",
        tenant_id="tenant-a",
        user_id="same-user",
        tokenizer=tokenizer,
    )
    tenant_b_chunks = chunk_blocks(
        blocks,
        document_id="same-document",
        tenant_id="tenant-b",
        user_id="same-user",
        tokenizer=tokenizer,
    )

    assert len(tenant_a_chunks) == 1
    assert len(tenant_b_chunks) == 1
    assert tenant_a_chunks[0].content_hash == tenant_b_chunks[0].content_hash
    assert tenant_a_chunks[0].chunk_id != tenant_b_chunks[0].chunk_id
