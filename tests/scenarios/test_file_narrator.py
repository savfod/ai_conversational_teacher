"""Tests for FileChunkLoader class."""

from pathlib import Path

import pytest

from conversa.generated.file_narrator import FileChunkLoader


def test_load_txt_file(tmp_path: Path):
    """Test loading a simple text file."""
    content = "Hello world. This is a test file."

    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content)

    loader = FileChunkLoader(file_path=str(temp_file), chunk_size=100)
    assert loader.text == content
    assert loader.text_length == len(content)


def test_chunk_splitting_basic(tmp_path: Path):
    """Test basic chunk splitting."""
    content = "First sentence. Second sentence. Third sentence."

    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content)

    loader = FileChunkLoader(file_path=str(temp_file), chunk_size=20)

    chunks = []
    while loader.has_more_chunks():
        chunk = loader.get_next_chunk()
        assert chunk is not None
        chunks.append(chunk.text)

    # Should have split into multiple chunks
    assert len(chunks) > 1
    # Verify all content is covered
    assert " ".join(chunks).replace("  ", " ") in content or content in " ".join(chunks)


def test_chunk_splitting_at_sentence_boundaries(tmp_path: Path):
    """Test that chunks prefer to split at sentence boundaries."""
    content = "First sentence. Second sentence. Third sentence. Fourth sentence."

    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content)

    loader = FileChunkLoader(file_path=str(temp_file), chunk_size=25)

    chunk = loader.get_next_chunk()
    assert chunk is not None

    # First chunk should end at a sentence boundary
    assert chunk.text.endswith(".")


def test_start_position(tmp_path: Path):
    """Test starting from a specific position."""
    content = "First sentence. Second sentence. Third sentence."

    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content)

    # Start from position 16 (after "First sentence. ")
    loader = FileChunkLoader(
        file_path=str(temp_file), start_position=16, chunk_size=100
    )

    chunk = loader.get_next_chunk()
    assert chunk is not None

    # Should start with "Second sentence"
    assert chunk.text.startswith("Second")
    assert "First" not in chunk.text


def test_file_not_found():
    """Test that FileNotFoundError is raised for non-existent files."""
    with pytest.raises(FileNotFoundError):
        FileChunkLoader(file_path="/nonexistent/file.txt")


def test_has_more_chunks(tmp_path: Path):
    """Test has_more_chunks method."""
    content = "Short text."

    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content)

    loader = FileChunkLoader(file_path=str(temp_file), chunk_size=100)

    assert loader.has_more_chunks() is True
    loader.get_next_chunk()
    assert loader.has_more_chunks() is False
