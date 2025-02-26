"""Tests for CLI argument parsing and ChatAnalysisOptions functionality."""

import os
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from cli import CLIParser
from chat_analysis_options import ChatAnalysisOptions

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

def test_cli_parser_defaults():
    """Test CLI parser with default values."""
    with patch('sys.argv', ['app.py']):
        args = CLIParser.parse_args()
        assert args.output == "analysis"
        assert args.pdf is None
        assert args.pdf_dir == "pdf_analysis"
        assert args.pdf_size_limit == 1.0
        assert args.date is None
        assert args.export_chat is None
        assert args.export_format == "txt"
        assert args.trends is None
        assert args.verify_format is False
        assert args.chat_id is None

def test_cli_parser_custom_values():
    """Test CLI parser with custom values."""
    test_date = "2024-01-01"
    with patch('sys.argv', [
        'app.py',
        '-o', 'custom_output',
        '--pdf', '3',
        '--pdf-dir', 'custom_pdfs',
        '--pdf-size-limit', '2.5',
        '-d', test_date,
        '--export-chat', 'chat123',
        '--export-format', 'json',
        '--trends', 'analysis_dir',
        '--verify-format',
        '--chat-id', 'chat456'
    ]):
        args = CLIParser.parse_args()
        assert args.output == "custom_output"
        assert args.pdf == 3
        assert args.pdf_dir == "custom_pdfs"
        assert args.pdf_size_limit == 2.5
        assert args.date == datetime.strptime(test_date, '%Y-%m-%d').date()
        assert args.export_chat == "chat123"
        assert args.export_format == "json"
        assert args.trends == "analysis_dir"
        assert args.verify_format is True
        assert args.chat_id == "chat456"

def test_verify_markdown_format(temp_dir, sample_args):
    """Test markdown file verification."""
    # Create test files
    os.makedirs(os.path.join(temp_dir, "analysis"), exist_ok=True)
    with open(os.path.join(temp_dir, "analysis", "valid.md"), "w") as f:
        f.write("# Valid markdown\n\n## Analysis\n\nContent")
    with open(os.path.join(temp_dir, "analysis", "invalid.md"), "w") as f:
        f.write("Invalid markdown")
    
    with patch('sys.argv', ['app.py', '-o', os.path.join(temp_dir, "analysis"), '--verify-format']), \
         patch('file_validator.FileValidator.verify_and_clean_md_files') as mock_verify:
        mock_verify.return_value = (["invalid.md"], 1)
        options = ChatAnalysisOptions()
        options.verify_markdown_format()
        mock_verify.assert_called_once_with(os.path.join(temp_dir, "analysis"))
