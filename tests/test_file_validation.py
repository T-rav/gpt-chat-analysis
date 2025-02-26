"""Tests for file validation functionality."""

import os
import pytest
from unittest.mock import MagicMock, patch, mock_open

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

def test_verify_md_format_valid_file():
    """Test verification of a valid markdown file with all required sections."""
    valid_content = '''
# 1. Brief Summary
Content here

# 2. Five-Step Decision Loop Analysis
## Step 1: Problem Framing & Initial Prompting
## Step 2: Response Evaluation & Validation
## Step 3: Expertise Application
## Step 4: Critical Assessment
### 4.1 Loop Completion Analysis
### 4.2 Breakdown Analysis
## Step 5: Process Improvement
# 3. Collaborative Pattern Analysis
## Observed Patterns
## Novel Patterns
# 4. Recommendations
'''
    with patch('builtins.open', mock_open(read_data=valid_content)):
        assert FileValidator.verify_md_format('test.md') is True

def test_verify_md_format_missing_sections():
    """Test verification of a markdown file with missing sections."""
    invalid_content = '''
# 1. Brief Summary
Content here
# 2. Five-Step Decision Loop Analysis
'''
    with patch('builtins.open', mock_open(read_data=invalid_content)):
        assert FileValidator.verify_md_format('test.md') is False

def test_verify_md_format_generic_content():
    """Test verification of a markdown file with generic placeholder content."""
    generic_content = '''
# 1. Brief Summary
The USER engaged with the AI in this conversation.
'''
    with patch('builtins.open', mock_open(read_data=generic_content)):
        assert FileValidator.verify_md_format('test.md') is False

def test_verify_md_format_debug_mode():
    """Test verification with debug mode enabled."""
    invalid_content = '''
# 1. Brief Summary
The USER engaged with the AI
'''
    with patch('builtins.open', mock_open(read_data=invalid_content)), \
         patch('builtins.print') as mock_print:
        assert FileValidator.verify_md_format('test.md', debug=True) is False
        mock_print.assert_called()

def test_verify_md_format_file_error():
    """Test verification when file reading raises an error."""
    with patch('builtins.open', side_effect=Exception('File error')):
        assert FileValidator.verify_md_format('nonexistent.md') is False

def test_verify_and_clean_md_files_nonexistent_dir():
    """Test verification with non-existent directory."""
    invalid_files, count = FileValidator.verify_and_clean_md_files('/nonexistent/dir')
    assert invalid_files == []
    assert count == 0

def test_verify_and_clean_md_files_mixed_content(temp_dir):
    """Test verification and cleaning with mixed valid and invalid files."""
    # Create test files
    valid_content = '# 1. Brief Summary\n' + all_required_sections()
    invalid_content = 'Invalid content'
    
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, 'valid.md'), 'w') as f:
        f.write(valid_content)
    with open(os.path.join(temp_dir, 'invalid.md'), 'w') as f:
        f.write(invalid_content)
    
    invalid_files, count = FileValidator.verify_and_clean_md_files(temp_dir)
    assert 'invalid.md' in invalid_files
    assert count == 1
    assert not os.path.exists(os.path.join(temp_dir, 'invalid.md'))
    assert os.path.exists(os.path.join(temp_dir, 'valid.md'))

def test_verify_and_clean_md_files_removal_error(temp_dir):
    """Test handling of file removal errors."""
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, 'test.md'), 'w') as f:
        f.write('Invalid content')
    
    with patch('os.remove', side_effect=Exception('Permission denied')):
        invalid_files, count = FileValidator.verify_and_clean_md_files(temp_dir)
        assert 'test.md' in invalid_files
        assert count == 1
        assert os.path.exists(os.path.join(temp_dir, 'test.md'))

def all_required_sections():
    """Helper function to generate all required sections."""
    return '''
# 2. Five-Step Decision Loop Analysis
## Step 1: Problem Framing & Initial Prompting
## Step 2: Response Evaluation & Validation
## Step 3: Expertise Application
## Step 4: Critical Assessment
### 4.1 Loop Completion Analysis
### 4.2 Breakdown Analysis
## Step 5: Process Improvement
# 3. Collaborative Pattern Analysis
## Observed Patterns
## Novel Patterns
# 4. Recommendations
'''
