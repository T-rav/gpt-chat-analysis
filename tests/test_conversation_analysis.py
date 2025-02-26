"""Tests for the conversation analysis functionality."""

import os
import pytest
from unittest.mock import MagicMock, patch

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
    messages = []
    
    # Create a file that already exists
    output_path = os.path.join(temp_dir, f"{chat_id}.md")
    with open(output_path, 'w') as f:
        f.write("Existing analysis")
    
    # Mock OpenAI (should not be called)
    mock_response = MagicMock()
    with patch.object(conversation_data.openai_client.chat.completions, 'create', return_value=mock_response) as mock_create:
        result_path, success = conversation_data.analyze_and_save_chat(chat_id, messages, temp_dir)
        
        # Verify the function returned early
        assert success is False
        assert result_path == output_path
        mock_create.assert_not_called()
        
        # Verify original content was preserved
        with open(output_path, 'r') as f:
            content = f.read()
            assert content == "Existing analysis"

def test_analyze_and_save_chat_too_long(conversation_data, temp_dir):
    """Test skipping analysis for chats that exceed token limit."""
    chat_id = "test_chat"
    # Create a very long message that would exceed token limit
    messages = [{"author": {"role": "user"}, "content": {"parts": ["x" * 100000]}}]
    
    # Mock OpenAI to raise an error
    mock_create = MagicMock(side_effect=Exception("Token limit exceeded"))
    with patch.object(conversation_data.openai_client.chat.completions, 'create', new=mock_create):
        output_path, success = conversation_data.analyze_and_save_chat(chat_id, messages, temp_dir)
        
        # Verify the function returned early due to token limit
        assert success is False
        assert output_path == os.path.join(temp_dir, f"{chat_id}.md")
        mock_create.assert_called_once()
        
        # Verify no file was created since analysis failed
        assert not os.path.exists(output_path)

def test_analyze_all_chats_parallel(conversation_data, temp_dir):
    """Test analyzing multiple chats in parallel."""
    chats = {
        "chat1": [{"author": {"role": "user"}, "content": {"parts": ["Message 1"]}}],
        "chat2": [{"author": {"role": "user"}, "content": {"parts": ["Message 2"]}}]
    }
    
    # Mock OpenAI responses
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Analysis"
    
    with patch.object(conversation_data.openai_client.chat.completions, 'create', return_value=mock_response):
        with patch.object(conversation_data, '_load_chat_data', return_value=chats):
            # Set research folder to temp_dir for testing
            conversation_data.config.research_folder = temp_dir
            conversation_data.analyze_all_chats_parallel()
            
            # Verify output files were created
            for chat_id in chats:
                output_path = os.path.join(temp_dir, f"{chat_id}.md")
                assert os.path.exists(output_path)
                with open(output_path, 'r') as f:
                    assert f.read() == "Analysis"
