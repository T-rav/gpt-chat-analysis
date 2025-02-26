"""Tests for trend analysis functionality."""

import os
import json
import pytest
from unittest.mock import MagicMock, patch

from chat_analysis_options import ChatAnalysisOptions
from trend_processor import TrendProcessor
from configuration import Config

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return str(tmp_path)

@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        return Config()

@pytest.fixture
def trend_processor(temp_dir):
    """Create a TrendProcessor instance with mocked OpenAI client."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        processor = TrendProcessor(output_dir=temp_dir)
        processor.model = "gpt-4"  # Ensure consistent model name
        return processor

def create_mock_completion(content):
    """Create a mock OpenAI completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response

def test_trend_processor_initialization(temp_dir):
    """Test TrendProcessor initialization and configuration."""
    # Test missing API key
    with patch.dict(os.environ, {'OPENAI_API_KEY': ''}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            TrendProcessor(output_dir=temp_dir)
    
    # Test successful initialization
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        processor = TrendProcessor(output_dir=temp_dir)
        assert os.path.exists(temp_dir)
        assert isinstance(processor.model, str)
        assert isinstance(processor.temperature, (int, float))

def test_analyze_directory_not_found(trend_processor):
    """Test handling of non-existent directory."""
    with pytest.raises(FileNotFoundError):
        trend_processor.analyze_directory("/nonexistent/path")

@pytest.mark.parametrize("response_content,expected_completed", [
    # Test valid JSON response
    ('{"loop_completion":{"completed":true,"exit_at_step_one":false,"skipped_validation":false},'
     '"breakdown":{"exit_step":"none","failure_reason":"none"},'
     '"insights":{"novel_patterns":true,"ai_partnership":true}}', True),
    # Test simple yes/no response
    ('yes', True),
    ('no', False),
    # Test invalid JSON response
    ('invalid json', False)
])
def test_analyze_with_openai(trend_processor, temp_dir, response_content, expected_completed):
    """Test OpenAI analysis with different response formats."""
    # Create test file
    test_file = "test.md"
    with open(os.path.join(temp_dir, test_file), "w") as f:
        f.write("# Test Content")
    
    # Mock OpenAI response
    mock_completion = create_mock_completion(response_content)
    
    with patch.object(trend_processor.client.chat.completions, 'create', return_value=mock_completion):
        result = trend_processor._analyze_with_openai("test content", test_file)
        assert isinstance(result, dict)
        assert 'loop_completion' in result
        assert result['loop_completion']['completed'] == expected_completed
        
        # Verify JSON file was created
        json_path = os.path.join(temp_dir, "test.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            saved_data = json.load(f)
            assert isinstance(saved_data, dict)
            assert 'loop_completion' in saved_data
            assert saved_data['loop_completion']['completed'] == expected_completed

def test_process_file(trend_processor, temp_dir):
    """Test processing of individual markdown files."""
    # Create test file with various content patterns
    test_content = """
# 1. Brief Summary
This is a test summary
# 2. Other Content
This is other content
"""
    
    test_file = os.path.join(temp_dir, "test.md")
    with open(test_file, "w") as f:
        f.write(test_content)
    
    mock_completion = create_mock_completion(
        '{"loop_completion":{"completed":true,"exit_at_step_one":false,"skipped_validation":false},'
        '"breakdown":{"exit_step":"none","failure_reason":"none"},'
        '"insights":{"novel_patterns":true,"ai_partnership":true}}'
    )
    
    with patch.object(trend_processor.client.chat.completions, 'create', return_value=mock_completion):
        result = trend_processor._process_file(test_file)
        assert isinstance(result, dict)
        assert result['completed'] == 1
        assert result['novel_patterns'] == 1
        assert result['exit_step'] == 'none'
        assert result['total'] == 1

def test_analyze_directory(trend_processor, temp_dir):
    """Test analysis of directory with multiple files."""
    # Create multiple test files
    for i in range(3):
        with open(os.path.join(temp_dir, f"test{i}.md"), "w") as f:
            f.write(f"# Test Chat {i}\nSome content")
    
    mock_completion = create_mock_completion(
        '{"loop_completion":{"completed":true,"exit_at_step_one":false,"skipped_validation":false},'
        '"breakdown":{"exit_step":"none","failure_reason":"none"},'
        '"insights":{"novel_patterns":true,"ai_partnership":true}}'
    )
    
    with patch.object(trend_processor.client.chat.completions, 'create', return_value=mock_completion):
        summary = trend_processor.analyze_directory(temp_dir)
        assert isinstance(summary, dict)
        assert 'Total Chats Analyzed' in summary
        assert summary['Total Chats Analyzed'] == 3
        assert summary['Loop Completion']['Completed (%)'] == 100.0
        assert summary['Insights']['Novel Patterns (%)'] == 100.0
        
        # Verify JSON files were created
        for i in range(3):
            json_path = os.path.join(temp_dir, f"test{i}.json")
            assert os.path.exists(json_path)
            with open(json_path) as f:
                data = json.load(f)
                assert isinstance(data, dict)
                assert 'loop_completion' in data
                assert data['loop_completion']['completed'] is True
