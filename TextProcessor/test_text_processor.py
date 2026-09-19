"""
Unit tests for the TextProcessor module.

Covers:
- MuffakirTextCleaner: all language modes, custom steps, empty input
- MuffakirChunking: orchestrator logic with default and injected chunkers
- ChunkingAndProcessing: load_documents, process_all, deprecated aliases
- chunkers/factory: all strategies, unknown strategy raises ValueError
"""
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from TextProcessor.MuffakirTextCleaner import MuffakirTextCleaner
from TextProcessor.MuffakirChunking import MuffakirChunking
from TextProcessor.ChunkingAndProcessing import ChunkingAndProcessing
from TextProcessor.chunkers import create_chunker, RecursiveChunker, FixedSizeChunker, SlidingWindowChunker, ContextualChunker


# ---------------------------------------------------------------------------
# MuffakirTextCleaner
# ---------------------------------------------------------------------------

class TestMuffakirTextCleaner:

    def test_arabic_clean_removes_non_arabic(self):
        cleaner = MuffakirTextCleaner(language="ar")
        text = "مرحبا بالعالم Hello World 123 @#$"
        result = cleaner.clean(text)
        # Arabic chars preserved, Latin preserved (mixed doc policy), symbols stripped
        assert "مرحبا" in result
        assert "Hello" in result
        assert "@" not in result
        assert "#" not in result

    def test_english_clean_removes_arabic(self):
        cleaner = MuffakirTextCleaner(language="en")
        text = "Hello World مرحبا 123"
        result = cleaner.clean(text)
        assert "Hello" in result
        assert "World" in result
        assert "مرحبا" not in result

    def test_auto_detect_arabic_doc(self):
        cleaner = MuffakirTextCleaner(language="auto")
        text = "مرحبا بالعالم العربي هذا نص طويل"
        result = cleaner.clean(text)
        assert "مرحبا" in result

    def test_auto_detect_english_doc(self):
        cleaner = MuffakirTextCleaner(language="auto")
        text = "This is a long English sentence with many words"
        result = cleaner.clean(text)
        assert "English" in result

    def test_none_language_passthrough(self):
        cleaner = MuffakirTextCleaner(language="none")
        text = "Raw text!!! @#$% مرحبا"
        result = cleaner.clean(text)
        assert result == text  # No modification

    def test_empty_text_returns_empty(self):
        cleaner = MuffakirTextCleaner(language="ar")
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""

    def test_unknown_language_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown language"):
            MuffakirTextCleaner(language="xyz")

    def test_add_and_remove_custom_step(self):
        cleaner = MuffakirTextCleaner(language="en")
        cleaner.add_cleaning_step("upper", lambda t: t.upper())
        result = cleaner.clean("hello world")
        assert result == result.upper()

        cleaner.remove_step("upper")
        result_after = cleaner.clean("hello world")
        assert result_after == result_after.lower()

    def test_custom_step_duplicate_replaced(self):
        cleaner = MuffakirTextCleaner(language="en")
        cleaner.add_cleaning_step("tag", lambda t: t + " [v1]")
        cleaner.add_cleaning_step("tag", lambda t: t + " [v2]")
        result = cleaner.clean("text")
        # Only one tag step should be applied
        assert result.count("[v") == 1
        assert "[v2]" in result

    def test_custom_step_exception_is_caught(self):
        """A failing custom step should not crash the pipeline."""
        cleaner = MuffakirTextCleaner(language="en")
        cleaner.add_cleaning_step("bad", lambda t: 1 / 0)
        # Should not raise
        result = cleaner.clean("hello")
        assert isinstance(result, str)

    def test_page_number_stripping(self):
        cleaner = MuffakirTextCleaner(language="ar")
        text = "بسم الله - 5 - الرحمن"
        result = cleaner.clean(text)
        assert "- 5 -" not in result

    def test_list_steps_does_not_raise(self):
        cleaner = MuffakirTextCleaner(language="ar")
        cleaner.add_cleaning_step("step1", lambda t: t)
        # Should log without raising
        cleaner.list_steps()


# ---------------------------------------------------------------------------
# MuffakirChunking
# ---------------------------------------------------------------------------

class TestMuffakirChunking:

    def test_default_recursive_chunker(self):
        orchestrator = MuffakirChunking()
        assert orchestrator.chunker.name == "recursive"

    def test_custom_chunker_string(self):
        orchestrator = MuffakirChunking(chunker="character", chunker_config={"size": 400, "overlap": 50})
        assert orchestrator.chunker.name == "fixed_size_character"

    def test_injected_chunker(self):
        custom = RecursiveChunker(size=200, overlap=50)
        orchestrator = MuffakirChunking(chunker=custom)
        assert orchestrator.chunker is custom

    def test_process_cleans_and_chunks(self):
        orchestrator = MuffakirChunking(chunker="recursive", language="en")
        docs = [Document(page_content="Hello world. " * 50, metadata={"source": "test.txt"})]
        result = orchestrator.process(docs)
        assert len(result) > 0
        assert all(isinstance(d, Document) for d in result)

    def test_process_skips_empty_after_clean(self):
        orchestrator = MuffakirChunking(chunker="recursive", language="ar")
        # A doc with only symbols that get stripped by Arabic cleaner
        docs = [Document(page_content="@@@@####$$$$$", metadata={})]
        result = orchestrator.process(docs)
        assert result == []

    def test_get_search_space_merges(self):
        orchestrator = MuffakirChunking()
        space = orchestrator.get_search_space()
        assert "chunker" in space
        assert "text_cleaner_language" in space

    def test_list_strategies_does_not_raise(self):
        orchestrator = MuffakirChunking()
        orchestrator.list_strategies()


# ---------------------------------------------------------------------------
# ChunkingAndProcessing
# ---------------------------------------------------------------------------

class TestChunkingAndProcessing:

    def test_load_documents_from_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample .txt file
            sample_file = os.path.join(tmpdir, "sample.txt")
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write("Hello world. This is a test document.")

            processor = ChunkingAndProcessing(directory_path=tmpdir)
            docs = processor.load_documents()
            assert len(docs) >= 1

    def test_load_documents_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkingAndProcessing(directory_path=tmpdir)
            docs = processor.load_documents()
            assert docs == []

    def test_load_documents_all_txt_files_fail_raises(self):
        from Muffakir.exceptions import ParsingError

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "sample.txt")
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write("Hello world.")

            processor = ChunkingAndProcessing(directory_path=tmpdir)
            with patch(
                "TextProcessor.ChunkingAndProcessing.TextLoader",
                side_effect=RuntimeError("boom"),
            ):
                with pytest.raises(ParsingError):
                    processor.load_documents()

    def test_load_documents_pdf_batch_failure_raises(self):
        from Muffakir.exceptions import ParsingError

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_file = os.path.join(tmpdir, "broken.pdf")
            with open(pdf_file, "wb") as f:
                f.write(b"%PDF-1.4\ninvalid")
            processor = ChunkingAndProcessing(directory_path=tmpdir)
            with patch(
                "TextProcessor.ChunkingAndProcessing.PdfReader",
                side_effect=RuntimeError("pdf backend missing"),
            ):
                with pytest.raises(ParsingError):
                    processor.load_documents()

    def test_load_documents_pdf_uses_page_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_file = os.path.join(tmpdir, "sample.pdf")
            with open(pdf_file, "wb") as f:
                f.write(b"%PDF-1.4\n")

            page = MagicMock()
            page.extract_text.return_value = "Text from a PDF page"
            reader = MagicMock(pages=[page])
            processor = ChunkingAndProcessing(directory_path=tmpdir)
            with patch(
                "TextProcessor.ChunkingAndProcessing.PdfReader",
                return_value=reader,
            ):
                docs = processor.load_documents()

            assert len(docs) == 1
            assert docs[0].page_content == "Text from a PDF page"
            assert docs[0].metadata["source"] == pdf_file
            assert docs[0].metadata["page"] == 0
            assert docs[0].metadata["page_label"] == "1"

    def test_process_all_no_ocr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "sample.txt")
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write("This is English text. " * 30)

            processor = ChunkingAndProcessing(
                directory_path=tmpdir,
                muffakir_chunking=MuffakirChunking(chunker="recursive", language="en"),
            )
            result = processor.process_all()
            assert len(result) >= 1
            # Verify chunk_id metadata is added
            assert result[0].metadata.get("chunk_id") == 1

    def test_process_all_ocr_requires_parser(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkingAndProcessing(directory_path=tmpdir)
            with pytest.raises(ValueError, match="document_parser must be provided"):
                processor.process_all(use_ocr=True)

    def test_process_all_ocr_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_parser = MagicMock()
            mock_pdoc = MagicMock()
            mock_pdoc.text = "مرحبا بالعالم " * 20
            mock_pdoc.source_path = os.path.join(tmpdir, "doc.pdf")
            mock_pdoc.parser_name = "mock_parser"
            mock_pdoc.metadata = {}
            mock_parser.parse_directory.return_value = [mock_pdoc]

            processor = ChunkingAndProcessing(
                directory_path=tmpdir,
                document_parser=mock_parser,
                muffakir_chunking=MuffakirChunking(chunker="recursive", language="ar"),
            )
            result = processor.process_all(use_ocr=True)
            assert len(result) >= 1
            assert result[0].metadata.get("ocr_processed") is True

    def test_process_all_legacy_chunking_method_override(self):
        """Passing a legacy chunking_method should trigger the override path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "doc.txt")
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write("English text here. " * 30)

            processor = ChunkingAndProcessing(
                directory_path=tmpdir,
                muffakir_chunking=MuffakirChunking(chunker="recursive", language="en"),
            )
            # Passing a different method should warn and override
            result = processor.process_all(chunking_method="character")
            assert len(result) >= 1

    def test_clean_text_static_alias(self):
        result = ChunkingAndProcessing.clean_text("Hello world 123", language="en")
        assert "Hello" in result

    def test_clean_arabic_text_static_alias(self):
        result = ChunkingAndProcessing.clean_arabic_text("مرحبا بالعالم")
        assert "مرحبا" in result


# ---------------------------------------------------------------------------
# Chunker Factory
# ---------------------------------------------------------------------------

class TestChunkerFactory:

    def test_create_recursive(self):
        chunker = create_chunker("recursive", size=500, overlap=100)
        assert chunker.name == "recursive"

    def test_create_character(self):
        chunker = create_chunker("character", size=300)
        assert chunker.name == "fixed_size_character"

    def test_create_token(self):
        chunker = create_chunker("token", size=256)
        assert chunker.name == "fixed_size_token"

    def test_create_sliding_window(self):
        chunker = create_chunker("sliding_window", window_size=400, step_size=200)
        assert chunker.name == "sliding_window"

    def test_create_contextual(self):
        chunker = create_chunker("contextual_recursive", size=400)
        assert "contextual" in chunker.name

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            create_chunker("not_a_strategy")

    def test_deprecated_spacy_falls_back(self):
        """spaCy strategy should fall back to recursive without raising."""
        chunker = create_chunker("spacy")
        assert chunker.name == "recursive"

    def test_chunk_produces_documents(self):
        chunker = RecursiveChunker(size=100, overlap=20)
        docs = [Document(page_content="word " * 200, metadata={"source": "test.txt"})]
        result = chunker.chunk(docs)
        assert len(result) > 1
        assert all(isinstance(d, Document) for d in result)

    def test_get_search_space_present(self):
        for strategy in ("recursive", "character", "token", "sliding_window"):
            chunker = create_chunker(strategy)
            space = chunker.get_search_space()
            assert isinstance(space, dict)
            assert len(space) > 0
