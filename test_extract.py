#!/usr/bin/env python3
"""Comprehensive tests for extract.py"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from extract import (
    clean_text,
    find_paper_metadata,
    load_paper_metadata,
    load_processed_papers,
    normalize_title,
    split_long_text,
)


# =============================================================================
# Unit Tests: Helper Functions
# =============================================================================

class TestNormalizeTitle:
    """Test title normalization for matching."""
    
    def test_basic_normalization(self):
        assert normalize_title("Hello World") == "hello_world"
    
    def test_special_characters_removed(self):
        assert normalize_title("DisCO: A Framework!") == "disco_a_framework"
        assert normalize_title("What's New?") == "whats_new"
    
    def test_multiple_spaces_collapsed(self):
        assert normalize_title("Hello   World") == "hello_world"
    
    def test_hyphens_preserved(self):
        assert normalize_title("Self-Attention Mechanism") == "self-attention_mechanism"
    
    def test_numbers_preserved(self):
        assert normalize_title("GPT-4 is Amazing") == "gpt-4_is_amazing"
    
    def test_empty_string(self):
        assert normalize_title("") == ""
    
    def test_whitespace_only(self):
        assert normalize_title("   ") == ""


class TestCleanText:
    """Test text cleaning for extracted content."""
    
    def test_whitespace_normalization(self):
        assert clean_text("Hello\n\nWorld") == "Hello World"
        assert clean_text("Hello\t\tWorld") == "Hello World"
    
    def test_citation_removal(self):
        assert clean_text("This is proven [1].") == "This is proven ."
        assert clean_text("See [1, 2, 3] for details.") == "See  for details."
        assert clean_text("Results [12] show [34, 56].") == "Results  show ."
    
    def test_preserves_content(self):
        text = "Machine learning is powerful."
        assert clean_text(text) == text
    
    def test_strips_whitespace(self):
        assert clean_text("  Hello World  ") == "Hello World"


class TestSplitLongText:
    """Test text splitting for token limits."""
    
    def test_short_text_unchanged(self):
        text = "This is a short text."
        result = split_long_text(text, max_chars=100)
        assert result == [text]
    
    def test_splits_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        result = split_long_text(text, max_chars=30)
        assert len(result) >= 2
        # Each chunk should end with a sentence
        for chunk in result[:-1]:
            assert chunk.endswith(".")
    
    def test_handles_very_long_sentence(self):
        text = "A" * 100  # Single "sentence" with no breaks
        result = split_long_text(text, max_chars=30)
        assert len(result) >= 1
        assert all(len(chunk) <= 100 for chunk in result)  # Should handle gracefully
    
    def test_empty_text(self):
        result = split_long_text("", max_chars=100)
        assert result == [""]
    
    def test_exact_boundary(self):
        text = "Hello."
        result = split_long_text(text, max_chars=6)
        assert result == ["Hello."]


# =============================================================================
# Unit Tests: CSV Metadata Loading
# =============================================================================

class TestLoadPaperMetadata:
    """Test CSV loading and metadata extraction."""
    
    def test_load_valid_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "title,decision,keywords,affiliations,openreview_url,semanticscholar_tldr\n"
            "Test Paper,poster,ml; ai,MIT; Stanford,https://openreview.net/1,A great paper.\n"
            "Another Paper,oral,nlp,Google,https://openreview.net/2,\n"
        )
        
        lookup = load_paper_metadata(str(csv_path))
        
        assert len(lookup) == 2
        assert "test_paper" in lookup
        assert "another_paper" in lookup
        
        meta = lookup["test_paper"]
        assert meta["title"] == "Test Paper"
        assert meta["decision"] == "poster"
        assert meta["keywords"] == "ml; ai"
        assert meta["affiliations"] == "MIT; Stanford"
        assert meta["url"] == "https://openreview.net/1"
        assert meta["tldr"] == "A great paper."
    
    def test_handles_missing_fields(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "title,decision\n"
            "Test Paper,poster\n"
        )
        
        lookup = load_paper_metadata(str(csv_path))
        meta = lookup["test_paper"]
        
        assert meta["title"] == "Test Paper"
        assert meta["decision"] == "poster"
        assert meta["keywords"] == ""
        assert meta["url"] == ""
    
    def test_handles_nan_values(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            "title": ["Test Paper"],
            "decision": ["poster"],
            "keywords": [None],
            "affiliations": [float("nan")],
            "openreview_url": ["https://test.com"],
            "semanticscholar_tldr": [None],
        })
        df.to_csv(csv_path, index=False)
        
        lookup = load_paper_metadata(str(csv_path))
        meta = lookup["test_paper"]
        
        assert meta["keywords"] == ""
        assert meta["affiliations"] == ""
        assert meta["tldr"] == ""
    
    def test_empty_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("title,decision\n")
        
        lookup = load_paper_metadata(str(csv_path))
        assert lookup == {}


class TestFindPaperMetadata:
    """Test metadata lookup by PDF stem."""
    
    @pytest.fixture
    def sample_lookup(self):
        return {
            "test_paper_on_machine_learning": {
                "title": "Test Paper on Machine Learning",
                "decision": "poster",
                "keywords": "ml",
                "affiliations": "MIT",
                "url": "https://test.com",
                "tldr": "A test paper.",
            },
            "another_paper": {
                "title": "Another Paper",
                "decision": "oral",
                "keywords": "nlp",
                "affiliations": "Google",
                "url": "https://test2.com",
                "tldr": "",
            },
        }
    
    def test_exact_match(self, sample_lookup):
        result = find_paper_metadata("Test_Paper_on_Machine_Learning", sample_lookup)
        assert result["title"] == "Test Paper on Machine Learning"
    
    def test_partial_match_truncated_filename(self, sample_lookup):
        # PDF filename might be truncated
        result = find_paper_metadata("Test_Paper_on_Machine", sample_lookup)
        assert result["title"] == "Test Paper on Machine Learning"
    
    def test_no_match_returns_fallback(self, sample_lookup):
        result = find_paper_metadata("Unknown_Paper_Title", sample_lookup)
        assert result["title"] == "Unknown Paper Title"
        assert "decision" not in result or result.get("decision") is None
    
    def test_handles_special_characters_in_stem(self, sample_lookup):
        result = find_paper_metadata("Test_Paper_on_Machine_Learning!", sample_lookup)
        # Should still match after normalization
        assert result["title"] == "Test Paper on Machine Learning"


# =============================================================================
# Unit Tests: Resume Functionality
# =============================================================================

class TestLoadProcessedPapers:
    """Test resume functionality."""
    
    def test_loads_processed_papers(self, tmp_path):
        output_path = tmp_path / "chunks.jsonl"
        output_path.write_text(
            '{"id": "paper1_abstract", "text": "...", "metadata": {"source_file": "paper1"}}\n'
            '{"id": "paper1_intro", "text": "...", "metadata": {"source_file": "paper1"}}\n'
            '{"id": "paper2_abstract", "text": "...", "metadata": {"source_file": "paper2"}}\n'
        )
        
        processed = load_processed_papers(str(output_path))
        
        assert processed == {"paper1", "paper2"}
    
    def test_empty_file(self, tmp_path):
        output_path = tmp_path / "chunks.jsonl"
        output_path.write_text("")
        
        processed = load_processed_papers(str(output_path))
        assert processed == set()
    
    def test_nonexistent_file(self, tmp_path):
        output_path = tmp_path / "nonexistent.jsonl"
        
        processed = load_processed_papers(str(output_path))
        assert processed == set()
    
    def test_handles_malformed_json(self, tmp_path):
        output_path = tmp_path / "chunks.jsonl"
        output_path.write_text(
            '{"id": "paper1_abstract", "metadata": {"source_file": "paper1"}}\n'
            'invalid json line\n'
            '{"id": "paper2_abstract", "metadata": {"source_file": "paper2"}}\n'
        )
        
        processed = load_processed_papers(str(output_path))
        assert processed == {"paper1", "paper2"}


# =============================================================================
# Integration Tests: Full Pipeline (with mocks)
# =============================================================================

class TestIntegrationMocked:
    """Integration tests with mocked external services."""
    
    @pytest.fixture
    def setup_test_env(self, tmp_path):
        """Set up test environment with CSV and PDF directory."""
        # Create CSV
        csv_path = tmp_path / "papers.csv"
        csv_path.write_text(
            "title,decision,keywords,affiliations,openreview_url,semanticscholar_tldr\n"
            "Test Paper One,poster,ml; deep learning,MIT,https://openreview.net/1,First paper tldr.\n"
            "Test Paper Two,oral,nlp; transformers,Google,https://openreview.net/2,Second paper tldr.\n"
        )
        
        # Create PDF directory
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        
        # Create dummy PDFs (just need to exist for the test)
        (pdf_dir / "Test_Paper_One.pdf").write_bytes(b"%PDF-1.4 dummy")
        (pdf_dir / "Test_Paper_Two.pdf").write_bytes(b"%PDF-1.4 dummy")
        
        # Output path
        output_path = tmp_path / "output.jsonl"
        
        return {
            "csv_path": csv_path,
            "pdf_dir": pdf_dir,
            "output_path": output_path,
            "tmp_path": tmp_path,
        }
    
    def test_full_pipeline_no_embed(self, setup_test_env):
        """Test full pipeline without embedding."""
        from extract import main
        import sys
        
        env = setup_test_env
        
        # Mock GROBID response
        mock_grobid_response = MagicMock()
        mock_grobid_response.status_code = 200
        mock_grobid_response.content = b'''<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
            <teiHeader/>
            <text>
                <front>
                    <abstract><p>This is the abstract of the paper.</p></abstract>
                </front>
                <body>
                    <div><head>Introduction</head><p>This is the introduction section with enough content to pass the 100 char threshold for testing purposes.</p></div>
                </body>
            </text>
        </TEI>'''
        
        mock_isalive = MagicMock()
        mock_isalive.status_code = 200
        
        with patch('extract.requests.post', return_value=mock_grobid_response):
            with patch('extract.requests.get', return_value=mock_isalive):
                with patch.object(sys, 'argv', [
                    'extract.py',
                    '--pdf-dir', str(env["pdf_dir"]),
                    '--output', str(env["output_path"]),
                    '--csv', str(env["csv_path"]),
                    '--no-embed',
                    '-n', '2',
                ]):
                    main()
        
        # Verify output
        assert env["output_path"].exists()
        
        chunks = []
        with open(env["output_path"]) as f:
            for line in f:
                chunks.append(json.loads(line))
        
        assert len(chunks) >= 2  # At least abstract + intro for each paper
        
        # Check metadata is populated
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            assert "title" in chunk["metadata"]
            assert "source_file" in chunk["metadata"]
            assert "embedding" not in chunk  # --no-embed flag


# =============================================================================
# Live Integration Tests (requires GROBID running)
# =============================================================================

class TestLiveIntegration:
    """Live tests that require GROBID. Skipped if GROBID not available."""
    
    @pytest.fixture
    def check_grobid(self):
        """Check if GROBID is running."""
        import requests
        try:
            r = requests.get("http://localhost:8070/api/isalive", timeout=2)
            if r.status_code != 200:
                pytest.skip("GROBID not running")
        except:
            pytest.skip("GROBID not running")
    
    @pytest.fixture
    def check_pdfs_exist(self):
        """Check if test PDFs exist."""
        pdf_dir = Path("NeurIPS_2025_PDFs")
        if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
            pytest.skip("No PDFs in NeurIPS_2025_PDFs/")
        return pdf_dir
    
    def test_extract_real_pdf(self, check_grobid, check_pdfs_exist, tmp_path):
        """Test extraction on a real PDF."""
        from extract import parse_grobid, load_paper_metadata, find_paper_metadata
        
        pdf_dir = check_pdfs_exist
        pdf_files = list(pdf_dir.glob("*.pdf"))[:1]
        pdf_path = pdf_files[0]
        
        # Parse with GROBID
        sections = parse_grobid(str(pdf_path), "http://localhost:8070")
        
        assert len(sections) > 0, "Should extract at least one section"
        
        # Check section structure
        for section in sections:
            assert "section" in section
            assert "content" in section
            assert len(section["content"]) > 0
        
        # Check we got an abstract
        section_names = [s["section"] for s in sections]
        assert "abstract" in section_names, "Should extract abstract"
        
        print(f"\nExtracted {len(sections)} sections from {pdf_path.name}:")
        for s in sections:
            print(f"  - {s['section']}: {len(s['content'])} chars")
    
    def test_metadata_matching(self, check_pdfs_exist):
        """Test that PDF filenames match CSV entries."""
        pdf_dir = check_pdfs_exist
        csv_path = Path("neurips_2025.csv")
        
        if not csv_path.exists():
            pytest.skip("neurips_2025.csv not found")
        
        lookup = load_paper_metadata(str(csv_path))
        pdf_files = list(pdf_dir.glob("*.pdf"))[:10]  # Test first 10
        
        matched = 0
        unmatched = []
        
        for pdf_path in pdf_files:
            metadata = find_paper_metadata(pdf_path.stem, lookup)
            if "decision" in metadata and metadata["decision"]:
                matched += 1
            else:
                unmatched.append(pdf_path.stem)
        
        print(f"\nMatched {matched}/{len(pdf_files)} PDFs to CSV entries")
        if unmatched:
            print(f"Unmatched: {unmatched[:5]}...")
        
        # At least 80% should match
        assert matched / len(pdf_files) >= 0.8, f"Too many unmatched PDFs: {unmatched}"
    
    def test_full_pipeline_live(self, check_grobid, check_pdfs_exist, tmp_path):
        """Full pipeline test with real PDFs (no embedding)."""
        from extract import main
        import sys
        
        pdf_dir = check_pdfs_exist
        output_path = tmp_path / "test_output.jsonl"
        
        with patch.object(sys, 'argv', [
            'extract.py',
            '--pdf-dir', str(pdf_dir),
            '--output', str(output_path),
            '--csv', 'neurips_2025.csv',
            '--no-embed',
            '-n', '3',  # Only process 3 papers
        ]):
            main()
        
        # Verify output
        assert output_path.exists()
        
        chunks = []
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        
        assert len(chunks) > 0, "Should produce at least one chunk"
        
        print(f"\nGenerated {len(chunks)} chunks from 3 papers:")
        
        # Group by source file
        by_source = {}
        for chunk in chunks:
            source = chunk["metadata"].get("source_file", "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk)
        
        for source, source_chunks in by_source.items():
            print(f"\n  {source}:")
            print(f"    Title: {source_chunks[0]['metadata'].get('title', 'N/A')}")
            print(f"    Decision: {source_chunks[0]['metadata'].get('decision', 'N/A')}")
            print(f"    Chunks: {len(source_chunks)}")
            for c in source_chunks[:3]:
                section = c["metadata"].get("section", c["id"].split("_")[-1])
                print(f"      - {section}: {len(c['text'])} chars")
        
        # Verify metadata fields
        sample = chunks[0]
        assert "title" in sample["metadata"]
        assert "source_file" in sample["metadata"]
        
        # If matched to CSV, should have these
        if sample["metadata"].get("decision"):
            assert "url" in sample["metadata"]
            assert "keywords" in sample["metadata"]


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
