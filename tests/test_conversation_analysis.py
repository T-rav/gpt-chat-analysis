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
            "content": {"content_type": "text", "parts": ["Hello, how can you help me?"]}
        },
        {
            "author": {"role": "assistant"},
            "content": {"content_type": "text", "parts": ["I can help you with your coding tasks."]}
        }
    ]
    
    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
# 1. Brief Summary
This chat shows a typical interaction.

# 2. Five-Step Decision Loop Analysis
## Step 1: Problem Framing & Initial Prompting
User clearly stated their need.

## Step 2: Response Evaluation & Validation
Assistant provided a clear response.

## Step 3: Expertise Application
Assistant demonstrated understanding.

## Step 4: Critical Assessment
### 4.1 Loop Completion Analysis
Loop was completed successfully.

### 4.2 Breakdown Analysis
No breakdown occurred.

## Step 5: Process Improvement
Process worked well.

# 3. Collaborative Pattern Analysis
## Observed Patterns
Good back-and-forth communication.

## Novel Patterns
None observed.

# 4. Recommendations
Continue with clear communication.
"""
    
    with patch.object(conversation_data.openai_client.chat.completions, 'create', return_value=mock_response):
        output_path, success = conversation_data.analyze_and_save_chat(chat_id, messages, temp_dir)
        
        # Verify success and output path
        assert success == 'success'
        assert output_path == os.path.join(temp_dir, f"{chat_id}.md")
        
        # Verify file was created with OpenAI's response
        with open(output_path, 'r') as f:
            content = f.read()
            assert "# 1. Brief Summary" in content
            assert "# 2. Five-Step Decision Loop Analysis" in content
            assert "# 3. Collaborative Pattern Analysis" in content
            assert "# 4. Recommendations" in content
        
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
    
    # Create a file that already exists with valid format
    output_path = os.path.join(temp_dir, f"{chat_id}.md")
    with open(output_path, 'w') as f:
        f.write("""
# 1. Brief Summary
Existing analysis

# 2. Five-Step Decision Loop Analysis
## Step 1: Problem Framing & Initial Prompting
Test

## Step 2: Response Evaluation & Validation
Test

## Step 3: Expertise Application
Test

## Step 4: Critical Assessment
### 4.1 Loop Completion Analysis
Test

### 4.2 Breakdown Analysis
Test

## Step 5: Process Improvement
Test

# 3. Collaborative Pattern Analysis
## Observed Patterns
Test

## Novel Patterns
Test

# 4. Recommendations
Test
""")
    
    # Mock OpenAI (should not be called)
    mock_response = MagicMock()
    with patch.object(conversation_data.openai_client.chat.completions, 'create', return_value=mock_response) as mock_create:
        result_path, success = conversation_data.analyze_and_save_chat(chat_id, messages, temp_dir)
        
        # Verify the function returned early
        assert success == 'skipped'
        assert result_path == output_path
        mock_create.assert_not_called()
        
        # Verify original content was preserved
        with open(output_path, 'r') as f:
            content = f.read()
            assert "# 1. Brief Summary" in content
            assert "Existing analysis" in content

def test_analyze_and_save_chat_too_long(conversation_data, temp_dir):
    """Test skipping analysis for chats that exceed token limit."""
    chat_id = "test_chat"
    # Create a very long message that would exceed token limit
    messages = [{"author": {"role": "user"}, "content": {"parts": ["x" * 100000]}}]
    
    # Mock OpenAI to raise an error
    mock_create = MagicMock(side_effect=Exception("Token limit exceeded"))
    with patch.object(conversation_data.openai_client.chat.completions, 'create', new=mock_create):
        output_path, success = conversation_data.analyze_and_save_chat(chat_id, messages, temp_dir)
        
        # Verify analysis was skipped due to API error
        assert success == 'api_error'
        assert output_path == os.path.join(temp_dir, f"{chat_id}.md")
        mock_create.assert_called_once()
        
        # Verify file was not created
        assert not os.path.exists(output_path)

def test_analyze_all_chats_parallel(conversation_data, temp_dir):
    """Test analyzing multiple chats in parallel."""
    chats = {
        "chat1": [{"author": {"role": "user"}, "content": {"parts": ["Message 1"]}}],
        "chat2": [{"author": {"role": "user"}, "content": {"parts": ["Message 2"]}}]
    }
    
    # Mock OpenAI responses with proper format
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
# 1. Brief Summary
Analysis content

# 2. Five-Step Decision Loop Analysis
## Step 1: Problem Framing & Initial Prompting
Test

## Step 2: Response Evaluation & Validation
Test

## Step 3: Expertise Application
Test

## Step 4: Critical Assessment
### 4.1 Loop Completion Analysis
Test

### 4.2 Breakdown Analysis
Test

## Step 5: Process Improvement
Test

# 3. Collaborative Pattern Analysis
## Observed Patterns
Test

## Novel Patterns
Test

# 4. Recommendations
Test
"""
    
    with patch.object(conversation_data.openai_client.chat.completions, 'create', return_value=mock_response):
        with patch.object(conversation_data, '_load_chat_data', return_value=chats):
            # Set research folder to temp_dir for testing
            conversation_data.config.research_folder = temp_dir
            conversation_data.analyze_all_chats_parallel()
            
            # Verify output files were created with proper format
            for chat_id in chats:
                output_path = os.path.join(temp_dir, f"{chat_id}.md")
                assert os.path.exists(output_path)
                with open(output_path, 'r') as f:
                    content = f.read()
                    assert "# 1. Brief Summary" in content
                    assert "# 2. Five-Step Decision Loop Analysis" in content
                    assert "# 3. Collaborative Pattern Analysis" in content
                    assert "# 4. Recommendations" in content
