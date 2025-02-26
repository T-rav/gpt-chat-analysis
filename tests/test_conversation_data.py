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

def test_export_chat_history_defaults_to_txt(conversation_data, sample_conversations, temp_dir):
    """Test that any non-JSON format (like markdown) defaults to TXT format."""
    chat_id = "test_chat"
    
    # Create necessary directories
    exports_dir = os.path.join(os.path.dirname(conversation_data.config.research_folder), 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    # Mock file operations
    mock_json = json.dumps(sample_conversations)
    with patch("builtins.open", mock_open(read_data=mock_json)) as mock_file:
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            
            # Any non-JSON format should create a TXT file
            output_path = conversation_data.export_chat_history(chat_id, "md")
            assert output_path.endswith(".txt")
            
            # Verify the file was opened for writing
            mock_file.assert_any_call(os.path.join(exports_dir, f"{chat_id}.txt"), "w")

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

def test_analyze_and_save_chat(conversation_data, temp_dir):
    """Test analyzing a chat with OpenAI and saving to markdown."""
    chat_id = "test_chat"
    messages = [
        {
            "author": {"role": "user"},
            "content": {"parts": ["Hello, how can you help me?"]}
        },
        {
            "author": {"role": "assistant"},
            "content": {"parts": ["I can help you with your coding tasks."]}
        }
    ]
    
    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "# Chat Analysis\n\nThis chat shows a typical interaction."
    
    with patch.object(conversation_data.openai_client.chat.completions, 'create', return_value=mock_response):
        output_path, success = conversation_data.analyze_and_save_chat(chat_id, messages, temp_dir)
        
        # Verify success and output path
        assert success is True
        assert output_path == os.path.join(temp_dir, f"{chat_id}.md")
        
        # Verify file was created with OpenAI's response
        with open(output_path, 'r') as f:
            content = f.read()
            assert content == "# Chat Analysis\n\nThis chat shows a typical interaction."
        
        # Verify OpenAI was called with correct parameters
        conversation_data.openai_client.chat.completions.create.assert_called_once()
        call_args = conversation_data.openai_client.chat.completions.create.call_args[1]
        assert call_args['model'] == conversation_data.config.model
        assert len(call_args['messages']) == 2
        assert call_args['messages'][0]['role'] == 'system'
        assert call_args['messages'][1]['role'] == 'user'
        assert 'Hello, how can you help me?' in call_args['messages'][1]['content']
        assert 'I can help you with your coding tasks.' in call_args['messages'][1]['content']

def test_analyze_and_save_chat_existing_file(conversation_data, temp_dir):
    """Test skipping analysis when file already exists."""
    chat_id = "test_chat"
    messages = [
        {
            "author": {"role": "user"},
            "content": {"parts": ["Hello"]}
        }
    ]
    
    # Create the output file first
    output_path = os.path.join(temp_dir, f"{chat_id}.md")
    os.makedirs(temp_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("Existing analysis")
    
    # Mock OpenAI (should not be called)
    mock_openai = MagicMock()
    with patch.object(conversation_data.openai_client.chat.completions, 'create', mock_openai):
        result_path, success = conversation_data.analyze_and_save_chat(chat_id, messages, temp_dir)
        
        # Verify no analysis was done
        assert success is False
        assert result_path == output_path
        mock_openai.assert_not_called()
        
        # Verify original content was not changed
        with open(output_path, 'r') as f:
            assert f.read() == "Existing analysis"

def test_analyze_and_save_chat_too_long(conversation_data, temp_dir):
    """Test skipping analysis for chats that exceed token limit."""
    chat_id = "test_chat"
    # Create a very long message that will exceed token limit
    messages = [
        {
            "author": {"role": "user"},
            "content": {"parts": ["Hello" * 50000]}
        }
    ]
    
    # Mock OpenAI (should not be called)
    mock_openai = MagicMock()
    with patch.object(conversation_data.openai_client.chat.completions, 'create', mock_openai):
        output_path, success = conversation_data.analyze_and_save_chat(chat_id, messages, temp_dir)
        
        # Verify no analysis was done
        assert success is False
        assert output_path == os.path.join(temp_dir, f"{chat_id}.md")
        mock_openai.assert_not_called()

def test_analyze_all_chats_parallel(conversation_data, sample_conversations, temp_dir):
    """Test analyzing multiple chats in parallel."""
    # Mock file operations for loading chats
    mock_json = json.dumps(sample_conversations)
    
    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "# Chat Analysis\n\nThis chat shows a typical interaction."
    
    with patch("builtins.open", mock_open(read_data=mock_json)) as mock_file, \
         patch("os.path.exists") as mock_exists, \
         patch.object(conversation_data.openai_client.chat.completions, 'create', return_value=mock_response):
        
        # Return True for conversations.json but False for analysis files
        def mock_exists_fn(path):
            return path.endswith('conversations.json')
        mock_exists.side_effect = mock_exists_fn
        
        conversation_data.config.research_folder = temp_dir
        
        # Run analysis
        conversation_data.analyze_all_chats_parallel()
        
        # Verify OpenAI was called for each chat
        assert conversation_data.openai_client.chat.completions.create.call_count == len(sample_conversations)
        
        # Verify output files were created
        for conv in sample_conversations:
            output_path = os.path.join(temp_dir, f"{conv['id']}.md")
            # We can't verify file existence since we're mocking os.path.exists
            # Instead verify that the file was opened for writing
            mock_file.assert_any_call(output_path, 'w')
