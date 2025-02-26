"""Tests for PDF generation functionality."""

import os
import pytest
from pathlib import Path
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

@pytest.fixture
def pdf_generator(temp_dir):
    """Create a PDFGenerator instance for testing."""
    markdown_dir = os.path.join(temp_dir, "markdown")
    output_dir = os.path.join(temp_dir, "pdf")
    os.makedirs(markdown_dir, exist_ok=True)
    return PDFGenerator(markdown_dir, output_dir)

@pytest.fixture
def sample_markdown_file(temp_dir):
    """Create a sample markdown file for testing."""
    markdown_dir = os.path.join(temp_dir, "markdown")
    os.makedirs(markdown_dir, exist_ok=True)
    file_path = os.path.join(markdown_dir, "test.md")
    with open(file_path, "w") as f:
        f.write("# Test Markdown\n\n## Section 1\n\nTest content")
    return file_path

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

def test_convert_markdown_to_pdf(pdf_generator, sample_markdown_file):
    """Test converting a single markdown file to PDF."""
    # Create a test markdown file
    with open(sample_markdown_file, 'w') as f:
        f.write('# Test')
    
    # Set up mocks
    mock_doc = MagicMock(spec=['write_pdf'])
    
    with patch('pdf_generator.HTML') as mock_html, \
         patch('pdf_generator.CSS') as mock_css, \
         patch('pdf_generator.FontConfiguration') as mock_font_config, \
         patch('pdf_generator.markdown2.markdown') as mock_markdown:
        
        # Configure mocks
        mock_markdown.return_value = '<h1>Test</h1>'
        mock_css_instance = MagicMock()
        mock_css.return_value = mock_css_instance
        
        # Configure HTML render to return our mock_doc
        mock_html_instance = MagicMock()
        mock_html_instance.render = MagicMock(return_value=mock_doc)
        mock_html.return_value = mock_html_instance
        
        output_path, success = pdf_generator.convert_markdown_to_pdf(
            Path(sample_markdown_file),
            "test.pdf"
        )
        
        # Print all calls made to our mocks
        print("\nMock calls:")
        print(f"HTML calls: {mock_html.mock_calls}")
        print(f"HTML instance calls: {mock_html_instance.mock_calls}")
        print(f"Doc calls: {mock_doc.mock_calls}")
        
        assert success is True
        assert output_path == Path(pdf_generator.output_dir) / "test.pdf"
        mock_doc.write_pdf.assert_called_once_with(
            target=output_path,
            zoom=0.9,
            optimize_images=True,
            jpeg_quality=70,
            compress=True,
            attachments=[]
        )

def test_convert_markdown_to_pdf_error_handling(pdf_generator):
    """Test error handling when converting markdown to PDF."""
    with patch('weasyprint.HTML') as mock_html:
        mock_html.side_effect = Exception("Test error")
        
        output_path, success = pdf_generator.convert_markdown_to_pdf(
            Path(pdf_generator.markdown_dir) / "nonexistent.md",
            "test.pdf"
        )
        
        assert success is False
        assert not output_path.exists()

def test_merge_markdown_files(pdf_generator, sample_markdown_file):
    """Test merging markdown files."""
    # Create additional test files
    files = [Path(sample_markdown_file)]
    for i in range(3):
        file_path = os.path.join(pdf_generator.markdown_dir, f"test{i}.md")
        with open(file_path, "w") as f:
            f.write(f"# Test {i}")
        files.append(Path(file_path))
    
    merged_files = pdf_generator.merge_markdown_files(files, target_chunks=2)
    assert len(merged_files) > 0
    assert all(f.suffix == ".md" for f in merged_files)

def test_generate_pdfs_empty_directory(pdf_generator):
    """Test generating PDFs when no markdown files exist."""
    # Don't create any markdown files
    pdf_generator.generate_pdfs(num_chunks=1)
    # Should not raise any errors

def test_generate_pdfs_with_files(pdf_generator):
    """Test generating PDFs with actual files."""
    # Create test files with UUID-like names
    for i in range(5):
        with open(os.path.join(pdf_generator.markdown_dir, f"test-{i}-uuid.md"), "w") as f:
            f.write(f"# Test {i}")
    
    # Create a mock PDF file
    mock_pdf = Path(pdf_generator.output_dir) / "test.pdf"
    mock_pdf.touch()
    
    with patch.object(pdf_generator, 'convert_all_markdown') as mock_convert:
        mock_convert.return_value = [mock_pdf]
        
        pdfs = pdf_generator.generate_pdfs(num_chunks=2)
        assert len(pdfs) == 1
        mock_convert.assert_called_once()

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



def test_convert_all_markdown(pdf_generator, sample_markdown_file):
    """Test converting all markdown files."""
    # Create test files with UUID-like names
    for i in range(3):
        file_path = os.path.join(pdf_generator.markdown_dir, f"test-{i}-uuid.md")
        with open(file_path, "w") as f:
            f.write(f"# Test {i}")
    
    with patch.object(pdf_generator, 'convert_markdown_to_pdf') as mock_convert, \
         patch.object(pdf_generator, 'merge_markdown_files') as mock_merge:
        mock_convert.return_value = (Path(pdf_generator.output_dir) / "test.pdf", True)
        mock_merge.return_value = [Path(pdf_generator.markdown_dir) / "merged.md"]
        
        pdfs = pdf_generator.convert_all_markdown()
        assert len(pdfs) == 1
        mock_convert.assert_called_once()

def test_merge_markdown_files_empty(pdf_generator):
    """Test merging markdown files with empty input."""
    merged = pdf_generator.merge_markdown_files([], target_chunks=1)
    assert len(merged) == 0
