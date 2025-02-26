"""Tests for file validation functionality."""

import os
import pytest
from unittest.mock import MagicMock, patch

from chat_analysis_options import ChatAnalysisOptions
from file_validator import FileValidator

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

def test_verify_markdown_format(temp_dir, sample_args):
    """Test markdown file verification."""
    # Create test files
    os.makedirs(os.path.join(temp_dir, "analysis"), exist_ok=True)
    with open(os.path.join(temp_dir, "analysis", "valid.md"), "w") as f:
        f.write("# Valid markdown")
    with open(os.path.join(temp_dir, "analysis", "invalid.md"), "w") as f:
        f.write("Invalid markdown without header")
    
    with patch('sys.argv', ['app.py', '-o', os.path.join(temp_dir, "analysis"), '--verify-format']), \
         patch('file_validator.FileValidator.verify_and_clean_md_files', return_value=(["invalid.md"], 1)):
        options = ChatAnalysisOptions()
        options.verify_markdown_format()

def test_verify_markdown_format_no_invalid_files(temp_dir, sample_args):
    """Test markdown file verification when all files are valid."""
    # Create test files
    os.makedirs(os.path.join(temp_dir, "analysis"), exist_ok=True)
    with open(os.path.join(temp_dir, "analysis", "valid1.md"), "w") as f:
        f.write("# Valid markdown 1")
    with open(os.path.join(temp_dir, "analysis", "valid2.md"), "w") as f:
        f.write("# Valid markdown 2")
    
    with patch('sys.argv', ['app.py', '-o', os.path.join(temp_dir, "analysis"), '--verify-format']), \
         patch('file_validator.FileValidator.verify_and_clean_md_files', return_value=([], 0)):
        options = ChatAnalysisOptions()
        options.verify_markdown_format()

def test_verify_markdown_format_error_handling(temp_dir, sample_args):
    """Test error handling during markdown file verification."""
    os.makedirs(os.path.join(temp_dir, "analysis"), exist_ok=True)
    
    with patch('sys.argv', ['app.py', '-o', os.path.join(temp_dir, "analysis"), '--verify-format']), \
         patch('file_validator.FileValidator.verify_and_clean_md_files', side_effect=Exception("Test error")):
        options = ChatAnalysisOptions()
        options.verify_markdown_format()  # Should handle the error gracefully
