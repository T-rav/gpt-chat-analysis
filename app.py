"""AI-powered chat conversation analyzer.

This module analyzes chat conversations using OpenAI's GPT models to evaluate:
1. Decision-making processes using a 5-step AI Decision Loop
2. Collaborative work patterns between humans and AI
3. Novel interaction patterns that emerge from the conversations
"""

# Standard library imports
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Union

# Third-party imports
import pytz
from openai import OpenAI
from tqdm import tqdm

# Local imports
from cli import CLIParser
from configuration import Config
from pdf_generator import PDFGenerator
from analysis_processor import AnalysisProcessor

class ConversationData:
    """Handles loading, processing, and analyzing chat conversations.
    
    This class manages the entire workflow of:
    1. Loading conversations from JSON files
    2. Processing timestamps and other metadata
    3. Analyzing conversations using OpenAI's GPT models
    4. Saving analysis results to markdown files
    """
    
    def __init__(self, config: Config) -> None:
        """Initialize the conversation data processor.
        
        Args:
            config: Configuration settings for processing
            
        Raises:
            ValueError: If OpenAI API key is missing
            FileNotFoundError: If conversation directory doesn't exist
        """
        self.config = config
        
        # Validate conversation directory
        if not os.path.exists(config.convo_folder):
            raise FileNotFoundError(f"Conversation directory not found: {config.convo_folder}")
        
        # Initialize OpenAI client
        try:
            self.openai_client = OpenAI(api_key=config.openai_api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")
        
        # Load and process conversations
        self.conversations = self._load_conversations()
        self.convo_times = self._process_timestamps()
        
    def export_chat_history(self, conversation_id: str, output_format: str = 'txt') -> str:
        """Export chat history for a specific conversation ID to a file.
        
        Args:
            conversation_id: ID of the conversation to export
            output_format: Format to export ('json', 'txt', or 'md')
            
        Returns:
            str: Path to the exported file
            
        Raises:
            ValueError: If conversation ID not found
        """
        # Find the conversation
        conversation = None
        for conv in self.conversations:
            # Check both 'id' and 'conversation_id' fields
            if conv.get('id') == conversation_id or conv.get('conversation_id') == conversation_id:
                conversation = conv
                break
                
        if not conversation:
            raise ValueError(f"Conversation ID not found: {conversation_id}")
            
        # Create output filename
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        output_dir = Path('exports')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"chat_{conversation_id}_{timestamp}.{output_format}"
        
        # Export based on format
        if output_format == 'json':
            with open(output_file, 'w') as f:
                json.dump(conversation, f, indent=2)
                
        elif output_format in ('txt', 'md'):
            with open(output_file, 'w') as f:
                f.write(f"Chat History for Conversation {conversation_id}\n")
                f.write(f"Exported at: {timestamp}\n")
                
                # Extract messages from mapping
                mapping = conversation.get('mapping', {})
                messages = []
                
                # Traverse the message tree from current node
                current_node = conversation.get('current_node')
                while current_node and current_node in mapping:
                    node = mapping[current_node]
                    message = node.get('message', {})
                    if message:
                        messages.insert(0, message)  # Insert at beginning to maintain order
                    current_node = node.get('parent')  # Move to parent node
                    
                # Write number of non-empty messages
                non_empty_count = len([m for m in messages 
                                     if m.get('content', {}).get('parts', [''])[0].strip() 
                                     and m.get('author', {}).get('role', '').strip()])
                f.write(f"Number of messages: {non_empty_count}\n\n")
                
                # Write messages in chronological order
                for msg in messages:
                    role = msg.get('author', {}).get('role', 'unknown')
                    content = msg.get('content', {}).get('parts', [''])[0]
                    
                    # Skip empty messages or messages with empty roles
                    if not content.strip() or not role.strip():
                        continue
                    
                    if output_format == 'md':
                        f.write(f"### {role.title()}\n\n{content}\n\n---\n\n")
                    else:  # txt
                        f.write(f"[{role.upper()}]\n{content}\n\n---\n")
                        
        print(f"\nExported chat history to: {output_file}")
        return str(output_file)

    def _load_conversations(self) -> List[Dict]:
        """Load conversations from JSON files with validation and error handling."""
        def validate_conversation(conv: Dict) -> bool:
            """Validate that a conversation has the required fields."""
            if not isinstance(conv, dict):
                print(f"Invalid conversation type: {type(conv)}")
                return False
                
            # Required fields for the conversation
            required_fields = ['id', 'conversation_id']
            missing_fields = [field for field in required_fields if field not in conv]
            if missing_fields:
                print(f"Missing required fields: {missing_fields}")
                return False
            
            return True

        def load_file(filepath: str) -> List[Dict]:
            """Load and validate a JSON file."""
            try:
                print(f"Loading conversations from {filepath}")
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        print(f"Error: Expected a list of conversations, got {type(data)}")
                        return []
                    
                    print(f"\nAnalyzing {len(data)} conversations...")
                    
                    # Filter out invalid conversations
                    valid_convs = []
                    for conv in data:
                        if validate_conversation(conv):
                            # Add empty messages list if not present
                            if 'messages' not in conv:
                                conv['messages'] = []
                            valid_convs.append(conv)
                    
                    print(f"\nLoaded {len(valid_convs)} valid conversations from {len(data)} total")
                    return valid_convs
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {filepath}: {str(e)}")
                return []
            except Exception as e:
                print(f"Error loading {filepath}: {str(e)}")
                return []

        # Try loading conversations.json first
        filepath = os.path.join(self.config.convo_folder, 'conversations.json')
        conversations = load_file(filepath)
        
        # If no valid conversations found, try shared_conversations.json
        if not conversations:
            filepath = os.path.join(self.config.convo_folder, 'shared_conversations.json')
            conversations = load_file(filepath)

        return conversations
    
    def analyze_and_save_chat(self, conv: Dict, research_folder: str) -> tuple[str, bool]:
        """Analyze a chat with OpenAI and save just the analysis to markdown.
        
        Args:
            conv: Dictionary containing the conversation data
            research_folder: Directory to save analysis results
            
        Returns:
            tuple[str, bool]: (Path to the output file, Whether file was processed or skipped)
            
        Note:
            If the output file already exists, the analysis will be skipped
            to support resuming interrupted analysis runs.
        """
        chat_id = conv.get('id', '')
        filename = f"{chat_id}.md"
        filepath = os.path.join(research_folder, filename)
        
        # Skip if file already exists (for resume support)
        if os.path.exists(filepath):
            return filepath, False
        
        # Create output directory if it doesn't exist
        os.makedirs(research_folder, exist_ok=True)
        
        # Extract messages from mapping
        mapping = conv.get('mapping', {})
        if not mapping:
            print(f"Warning: No mapping found in chat {chat_id}")
            return filepath, False  # Return tuple for consistency
            
        # Convert mapping to ordered list of messages
        messages = []
        current_node = conv.get('current_node')
        while current_node and current_node in mapping:
            node = mapping[current_node]
            message = node.get('message', {})
            if message:
                messages.insert(0, message)  # Insert at beginning to maintain order
            current_node = node.get('parent')  # Move to parent node
        
        if not messages:
            print(f"Warning: No messages found in chat {chat_id}")
            return filepath, False  # Return tuple for consistency
        
        # Check total conversation length and skip if too large
        total_chars = sum(len(msg.get('content', {}).get('parts', [''])[0]) for msg in messages)
        # Rough estimate: 1 char ≈ 0.25 tokens
        estimated_tokens = total_chars / 4
        
        if estimated_tokens > 100000:  # Conservative limit to avoid API errors
            print(f"Skipping chat {chat_id} - estimated {int(estimated_tokens)} tokens exceeds limit")
            return filepath, False
            
        # Prepare conversation for analysis
        conversation = ""
        for msg in messages:
            role = msg.get('author', {}).get('role', 'unknown')
            content = msg.get('content', {}).get('parts', [''])[0]
            conversation += f"{role}: {content}\n"
        
        # Analyze with OpenAI
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": conversation}
                ],
                temperature=self.config.temperature
            )
            
            analysis = response.choices[0].message.content
            
            # Write analysis to file only if API call succeeds
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(analysis)
            return filepath, True  # Successfully processed
            
        except Exception as e:
            print(f"\nError analyzing chat {chat_id}: {str(e)}")
            # Don't write error file, just return as skipped
            return filepath, False  # Count as skipped when API fails

    def analyze_single_chat(self, chat_id: str) -> None:
        """Analyze a specific chat and save the analysis as a markdown file."""
        for conv in self.conversations:
            if conv.get('id') == chat_id:
                filepath = self.analyze_and_save_chat(conv, self.config.research_folder)
                print(f"Analysis saved to {filepath}")
                return
        
        print(f"Chat with ID {chat_id} not found")

    def analyze_all_chats_parallel(self) -> None:
        """Analyze all chats with OpenAI and save results as markdown files using threading."""
        # Create research folder if it doesn't exist
        os.makedirs(self.config.research_folder, exist_ok=True)
        
        # Filter conversations by start date if specified
        filtered_conversations = self.conversations
        if self.config.start_date:
            original_count = len(filtered_conversations)
            filtered_conversations = [
                conv for conv in filtered_conversations
                if self.convo_times[conv['id']].date() >= self.config.start_date
            ]
            print(f"\nFiltered from {original_count} to {len(filtered_conversations)} conversations after {self.config.start_date}")
        
        total_count = len(filtered_conversations)
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        print(f"\nStarting analysis of {total_count} conversations...")
        
        # Use ThreadPoolExecutor with progress bar
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for conv in filtered_conversations:
                future = executor.submit(self.analyze_and_save_chat, conv, self.config.research_folder)
                futures.append(future)
            
            # Show progress bar while waiting for results
            futures_set = set(futures)  # Convert to set for O(1) lookup
            with tqdm(total=total_count, desc="Analyzing") as pbar:
                while futures_set:
                    # Update progress for any completed futures
                    done, _ = wait(futures_set, timeout=0.1)
                    for future in done:
                        try:
                            filepath, was_processed = future.result()
                            filename = os.path.basename(filepath)
                            if was_processed:
                                processed_count += 1
                                status = "Processing"
                            else:
                                skipped_count += 1
                                status = "Skipped"
                            pbar.set_description(f"{status}: {filename} (New: {processed_count}, Skip: {skipped_count})")
                            pbar.update(1)
                            futures_set.remove(future)
                        except Exception as e:
                            error_count += 1
                            pbar.write(f"Error processing conversation: {str(e)}")
                            pbar.update(1)
                            futures_set.remove(future)
        
        # Print summary
        print(f"\nAnalysis Summary")
        print(f"---------------")
        print(f"Total conversations: {total_count}")
        print(f"Newly processed:    {processed_count}")
        print(f"Already analyzed:   {skipped_count}")
        if error_count > 0:
            print(f"Errors:            {error_count}")
    
    def _process_timestamps(self) -> Dict[str, datetime]:
        """Process conversation timestamps into datetime objects.
        
        Returns:
            Dict[str, datetime]: Mapping of conversation IDs to their timestamps
        """
        return {
            str(conv.get('id')): datetime.fromtimestamp(conv.get('create_time', 0), tz=timezone.utc)
            .astimezone(pytz.timezone(self.config.local_tz))
            for conv in self.conversations
            if conv.get('id') and conv.get('create_time')
        }
        
from chat_analysis_app import ChatAnalysisApp

def main() -> None:
    """Entry point for the chat analysis tool."""
    app = ChatAnalysisApp()
    app.run()

if __name__ == '__main__':
    main()
