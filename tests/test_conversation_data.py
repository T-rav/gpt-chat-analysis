"""Tests for the ConversationData class."""

import json
import os
import pytest
from datetime import datetime
from unittest.mock import mock_open, patch, MagicMock

from configuration import Config
from conversation_data import ConversationData

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return str(tmp_path)

@pytest.fixture
def config(temp_dir):
    """Create a test configuration."""
    config = Config()
    config.convo_folder = os.path.join(temp_dir, "conversations")
    config.research_folder = os.path.join(temp_dir, "research")
    return config

@pytest.fixture
def conversation_data(config):
    """Create a ConversationData instance for testing."""
    return ConversationData(config)

@pytest.fixture
def sample_conversations():
    """Create sample conversations data for testing."""
    return [
        {
            "id": "test_chat",
            "create_time": 1000,
            "messages": [
                {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Hello"]}
                },
                {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Hi there"]}
                }
            ]
        }
    ]

def test_init(conversation_data, config):
    """Test ConversationData initialization."""
    assert conversation_data.config == config
    assert conversation_data.openai_client is not None

def test_export_chat_history_json(conversation_data, sample_conversations, temp_dir):
    """Test exporting chat history to JSON."""
    chat_id = "test_chat"
    
    # Create necessary directories
    exports_dir = os.path.join(os.path.dirname(conversation_data.config.research_folder), 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    # Mock file operations
    mock_json = json.dumps(sample_conversations)
    with patch("builtins.open", mock_open(read_data=mock_json)) as mock_file:
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            
            output_path = conversation_data.export_chat_history(chat_id, "json")
            
            # Check that the correct files were opened
            mock_file.assert_any_call(os.path.join(conversation_data.config.convo_folder, "conversations.json"), "r")
            mock_file.assert_any_call(os.path.join(exports_dir, f"{chat_id}.json"), "w")
            
            assert output_path == os.path.join(exports_dir, f"{chat_id}.json")

def test_export_chat_history_txt(conversation_data, sample_conversations, temp_dir):
    """Test exporting chat history to TXT format."""
    chat_id = "test_chat"
    
    # Create necessary directories
    exports_dir = os.path.join(os.path.dirname(conversation_data.config.research_folder), 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    # Mock file operations
    mock_json = json.dumps(sample_conversations)
    with patch("builtins.open", mock_open(read_data=mock_json)) as mock_file:
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            
            output_path = conversation_data.export_chat_history(chat_id, "txt")
            
            # Check that the correct files were opened
            mock_file.assert_any_call(os.path.join(conversation_data.config.convo_folder, "conversations.json"), "r")
            mock_file.assert_any_call(os.path.join(exports_dir, f"{chat_id}.txt"), "w")
            
            assert output_path == os.path.join(exports_dir, f"{chat_id}.txt")

def test_export_chat_history_invalid_chat(conversation_data, sample_conversations, temp_dir):
    """Test exporting non-existent chat."""
    # Create necessary directories
    exports_dir = os.path.join(os.path.dirname(conversation_data.config.research_folder), 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    # Mock file operations
    mock_json = json.dumps(sample_conversations)
    with patch("builtins.open", mock_open(read_data=mock_json)) as mock_file:
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            
            with pytest.raises(ValueError, match="Chat invalid_chat not found"):
                conversation_data.export_chat_history("invalid_chat", "json")

def test_export_chat_history_invalid_format(conversation_data, sample_conversations, temp_dir):
    """Test exporting chat with invalid format."""
    chat_id = "test_chat"
    
    # Create necessary directories
    exports_dir = os.path.join(os.path.dirname(conversation_data.config.research_folder), 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    # Mock file operations
    mock_json = json.dumps(sample_conversations)
    with patch("builtins.open", mock_open(read_data=mock_json)) as mock_file:
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            
            with pytest.raises(ValueError, match="Invalid export format: pdf. Must be one of: json, txt, md"):
                conversation_data.export_chat_history(chat_id, "pdf")

def test_load_chat_data(conversation_data, sample_conversations):
    """Test loading chat data."""
    # Mock file operations
    mock_json = json.dumps(sample_conversations)
    with patch("builtins.open", mock_open(read_data=mock_json)) as mock_file:
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            
            # Call the method directly on the instance
            chat_data = conversation_data._load_chat_data()
            
            # Verify we got the expected data
            assert len(chat_data) == 1
            assert "test_chat" in chat_data
            messages = chat_data["test_chat"]
            assert len(messages) == 2
            
            # Check that the correct file was opened
            mock_file.assert_called_once_with(
                os.path.join(conversation_data.config.convo_folder, "conversations.json"), "r"
            )
