"""Tests for PDF generation functionality."""

import os
import pytest
from unittest.mock import MagicMock, patch

from chat_analysis_options import ChatAnalysisOptions
from pdf_generator import PDFGenerator

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return str(tmp_path)

@pytest.fixture
def sample_args():
    """Create sample command line arguments."""
    args = MagicMock()
    args.output = "analysis"
    args.pdf = None
    args.pdf_dir = "pdf_analysis"
    args.pdf_size_limit = 1.0
    args.date = None
    args.export_chat = None
    args.export_format = "txt"
    args.trends = None
    args.verify_format = False
    args.chat_id = None
    return args

def test_generate_pdfs(temp_dir, sample_args):
    """Test PDF generation."""
    # Create test markdown files
    os.makedirs(os.path.join(temp_dir, "analysis"), exist_ok=True)
    with open(os.path.join(temp_dir, "analysis", "test1.md"), "w") as f:
        f.write("# Test Chat 1")
    
    with patch('sys.argv', ['app.py', '-o', os.path.join(temp_dir, "analysis"), '--pdf', '2']), \
         patch('pdf_generator.PDFGenerator.generate_pdfs') as mock_generate:
        options = ChatAnalysisOptions()
        options.generate_pdfs()
        mock_generate.assert_called_once_with(2)

@pytest.mark.parametrize("num_chunks,size_limit", [
    (1, 1.0),  # Single PDF with default size
    (3, 2.5),  # Multiple PDFs with custom size
])
def test_pdf_generator_options(temp_dir, sample_args, num_chunks, size_limit):
    """Test PDF generator with different options."""
    # Create test markdown files
    analysis_dir = os.path.join(temp_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    for i in range(3):
        with open(os.path.join(analysis_dir, f"test{i+1}.md"), "w") as f:
            f.write(f"# Test Chat {i+1}")
    
    # Mock the generate_pdfs method
    mock_generate = MagicMock()
    mock_pdf_gen = MagicMock()
    mock_pdf_gen.generate_pdfs = mock_generate
    
    with patch('sys.argv', [
        'app.py', 
        '-o', analysis_dir,
        '--pdf', str(num_chunks),
        '--pdf-dir', os.path.join(temp_dir, "custom_pdfs"),
        '--pdf-size-limit', str(size_limit)
    ]), patch('chat_analysis_options.PDFGenerator', return_value=mock_pdf_gen):
        options = ChatAnalysisOptions()
        options.generate_pdfs()
        mock_generate.assert_called_once_with(num_chunks)
