"""Tests for trend analysis functionality."""

import os
import pytest
from unittest.mock import MagicMock, patch

from chat_analysis_options import ChatAnalysisOptions
from trend_processor import TrendProcessor

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

@pytest.mark.parametrize("chat_id,num_files", [
    (None, 2),  # Test full directory
    ("test_chat", 1)  # Test single chat
])
def test_analyze_trends(temp_dir, sample_args, chat_id, num_files):
    """Test analyzing trends in a directory or single chat."""
    # Create test files
    os.makedirs(os.path.join(temp_dir, "analysis"), exist_ok=True)
    
    if chat_id:
        # Single chat case
        with open(os.path.join(temp_dir, "analysis", f"{chat_id}.md"), "w") as f:
            f.write("# Test Chat")
    else:
        # Directory case
        for i in range(num_files):
            with open(os.path.join(temp_dir, "analysis", f"test{i+1}.md"), "w") as f:
                f.write(f"# Test Chat {i+1}")
    
    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"loop_completion": {"completed": true}}'
    
    # Prepare CLI args
    argv = ['app.py', '--trends', os.path.join(temp_dir, "analysis")]
    if chat_id:
        argv.extend(['--chat-id', chat_id])
    
    with patch('sys.argv', argv), \
         patch('trend_processor.TrendProcessor._analyze_with_openai', return_value=mock_response) as mock_analyze:
        options = ChatAnalysisOptions()
        options.analyze_trends()
        
        # Verify analysis was performed
        assert mock_analyze.call_count > 0
