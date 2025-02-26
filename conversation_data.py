"""Handles loading and processing of conversation data."""

import concurrent.futures
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from openai import OpenAI

from configuration import Config
from pdf_generator import PDFGenerator

class ConversationData:
    """Handles loading and processing of conversation data."""

    def __init__(self, config: Config):
        """Initialize conversation data processor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.openai_client = OpenAI()

    def export_chat_history(self, chat_id: str, export_format: str) -> str:
        """Export a chat conversation to the specified format.
        
        Args:
            chat_id: ID of the chat to export
            export_format: Format to export to (json, md, or txt)
            
        Returns:
            Path to the exported file
        """
        # Find the raw chat file
        chat_file = os.path.join(self.config.convo_folder, f"{chat_id}.json")
        if not os.path.exists(chat_file):
            raise ValueError(f"Chat {chat_id} not found in {self.config.convo_folder}")
            
        # Read the source file
        with open(chat_file, 'r') as f:
            data = json.load(f)
            
        # Get messages in order
        messages = list(data['mapping'].values())
        messages.sort(key=lambda x: x.get('create_time', 0))
        
        if export_format == 'json':
            # Copy to research folder
            output_file = os.path.join(self.config.research_folder, f"{chat_id}.json")
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif export_format == 'md':
            # Convert to markdown with headers
            md_content = []
            for msg in messages:
                role = msg.get('role', '')
                if role == 'user':
                    md_content.append('# Human:')
                elif role == 'assistant':
                    md_content.append('# Assistant:')
                md_content.append(msg.get('content', ''))
                md_content.append('')  # Empty line between messages
            
            output_file = os.path.join(self.config.research_folder, f"{chat_id}.md")
            with open(output_file, 'w') as f:
                f.write('\n'.join(md_content))
                
        elif export_format == 'txt':
            # Convert to plain text without headers
            txt_content = []
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '').strip()
                if role == 'user':
                    txt_content.append(f"Human: {content}")
                elif role == 'assistant':
                    txt_content.append(f"Assistant: {content}")
                txt_content.append('')  # Empty line between messages
            
            output_file = os.path.join(self.config.research_folder, f"{chat_id}.txt")
            with open(output_file, 'w') as f:
                f.write('\n'.join(txt_content))
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
            
        return output_file

    def export_chat_history(self, chat_id: str, export_format: str) -> str:
        """Export a chat conversation to the specified format.
        
        Args:
            chat_id: ID of the chat to export
            export_format: Format to export to (json, md, or txt)
            
        Returns:
            Path to the exported file
        """
        # Find the chat directory
        chat_dir = os.path.join(self.config.convo_folder, chat_id)
        if not os.path.exists(chat_dir):
            raise ValueError(f"Chat directory not found: {chat_dir}")
            
        # Load chat data
        chat_file = os.path.join(chat_dir, 'chat.json')
        if not os.path.exists(chat_file):
            raise ValueError(f"Chat file not found: {chat_file}")
            
        with open(chat_file, 'r') as f:
            chat_data = json.load(f)
            
        # Get messages in order
        messages = []
        for msg in chat_data.get('messages', []):
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role and content:
                messages.append({'role': role, 'content': content})
        
        # Export in requested format
        if export_format == 'json':
            output_file = os.path.join(self.config.research_folder, f"{chat_id}.json")
            with open(output_file, 'w') as f:
                json.dump(chat_data, f, indent=2)
                
        elif export_format == 'md':
            md_content = []
            for msg in messages:
                role = msg['role']
                if role == 'user':
                    md_content.append('# Human:')
                elif role == 'assistant':
                    md_content.append('# Assistant:')
                md_content.append(msg['content'])
                md_content.append('')  # Empty line between messages
            
            output_file = os.path.join(self.config.research_folder, f"{chat_id}.md")
            with open(output_file, 'w') as f:
                f.write('\n'.join(md_content))
                
        elif export_format == 'txt':
            txt_content = []
            for msg in messages:
                role = msg['role']
                content = msg['content'].strip()
                if role == 'user':
                    txt_content.append(f"Human: {content}")
                elif role == 'assistant':
                    txt_content.append(f"Assistant: {content}")
                txt_content.append('')  # Empty line between messages
            
            output_file = os.path.join(self.config.research_folder, f"{chat_id}.txt")
            with open(output_file, 'w') as f:
                f.write('\n'.join(txt_content))
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
            
        return output_file

    def analyze_and_save_chat(self, chat_id: str, messages: List[Dict[str, Any]], 
                            output_dir: str) -> Tuple[str, bool]:
        """Analyze a chat conversation and save the results to a markdown file.
        
        Args:
            chat_id: ID of the chat to analyze
            messages: List of chat messages
            output_dir: Directory to save analysis results
            
        Returns:
            Tuple of (output filepath, success status)
        """
        filepath = os.path.join(output_dir, f"{chat_id}.md")
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"Skipping chat {chat_id} - analysis already exists")
            return filepath, False
            
        # Estimate token count (rough heuristic)
        estimated_tokens = sum(len(str(msg)) / 4 for msg in messages)
        if estimated_tokens > 32000:  # OpenAI's max context
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
            
            # Save analysis to markdown file
            analysis = response.choices[0].message.content
            os.makedirs(output_dir, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(analysis)
            return filepath, True
            
        except Exception as e:
            print(f"Error analyzing chat {chat_id}: {str(e)}")
            return filepath, False

    def analyze_all_chats_parallel(self) -> None:
        """Analyze all chat conversations in parallel."""
        chats = self._load_chat_data()
        if not chats:
            print("No chat data found")
            return

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for chat_id, messages in chats.items():
                future = executor.submit(
                    self.analyze_and_save_chat,
                    chat_id,
                    messages,
                    self.config.research_folder
                )
                futures.append(future)

            # Process results as they complete
            completed = 0
            successful = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                if future.result()[1]:
                    successful += 1
                print(f"\rProgress: {completed}/{len(futures)} chats processed", end='')
            print(f"\nCompleted! {successful}/{len(futures)} chats analyzed successfully")

        # Generate PDFs if requested
        if self.config.pdf_chunks:
            print(f"\nGenerating {self.config.pdf_chunks} PDF files...")
            pdf_gen = PDFGenerator(
                markdown_dir=self.config.research_folder,
                output_dir=self.config.pdf_output_dir,
                size_limit_mb=self.config.pdf_size_limit_mb
            )
            pdf_gen.generate_pdfs(self.config.pdf_chunks)

    def export_chat_history(self, chat_id: str, format: str = 'json') -> str:
        """Export a chat conversation to a file.
        
        Args:
            chat_id: ID of the chat to export
            format: Output format ('json', 'txt', or 'md')
            
        Returns:
            Path to the exported file
            
        Raises:
            ValueError: If format is not one of 'json', 'txt', or 'md'
        """
        # Validate format
        if format not in ['json', 'txt', 'md']:
            raise ValueError(f"Invalid export format: {format}. Must be one of: json, txt, md")
        # Create exports directory if it doesn't exist
        exports_dir = os.path.join(os.path.dirname(self.config.research_folder), 'exports')
        os.makedirs(exports_dir, exist_ok=True)
        
        # Load the conversation file
        conversations_file = os.path.join(self.config.convo_folder, 'conversations.json')
        if not os.path.exists(conversations_file):
            raise ValueError(f"No conversations.json found in {self.config.convo_folder}")
            
        with open(conversations_file, 'r') as f:
            conversations = json.load(f)
            
        print(f"Found {len(conversations)} conversations")
        # Find the target conversation
        target_conv = None
        for conv in conversations:
            if isinstance(conv, dict):
                conv_id = conv.get('id')
                print(f"Checking conversation {conv_id}")
                if conv_id == chat_id:
                    target_conv = conv
                    print(f"Found target conversation with ID {chat_id}")
                    break
                
        if not target_conv:
            raise ValueError(f"Chat {chat_id} not found")
            
        # Export in requested format
        if format == 'json':
            output_file = os.path.join(exports_dir, f"{chat_id}.json")
            with open(output_file, 'w') as f:
                json.dump(target_conv, f, indent=2)
        else:
            output_file = os.path.join(exports_dir, f"{chat_id}.txt")
            with open(output_file, 'w') as f:
                mapping = target_conv.get('mapping', {})
                print(f"Found {len(mapping)} messages in mapping")
                for msg_id, msg_data in mapping.items():
                    print(f"\nProcessing message {msg_id}")
                    print(f"Full message data: {msg_data}")
                    if isinstance(msg_data, dict) and 'message' in msg_data:
                        message = msg_data['message']
                        if isinstance(message, dict):
                            content = message.get('content')
                            role = message.get('author', {}).get('role', 'unknown')
                            print(f"Role: {role}")
                            
                            if isinstance(content, dict):
                                content_type = content.get('content_type')
                                print(f"Content type: {content_type}")
                                text_parts = []
                                if content_type == 'text':
                                    parts = content.get('parts', [])
                                    print(f"Text parts: {parts}")
                                    if parts and parts[0]:
                                        text_parts.append(parts[0])
                                elif content_type == 'multimodal_text':
                                    parts = content.get('parts', [])
                                    print(f"Multimodal parts: {parts}")
                                    for part in parts:
                                        print(f"Processing part: {part}")
                                        if isinstance(part, dict):
                                            part_type = part.get('content_type')
                                            print(f"Part type: {part_type}")
                                            if part_type == 'audio_transcription':
                                                text = part.get('text')
                                                print(f"Found text: {text}")
                                                if text:
                                                    text_parts.append(text)
                                            elif part_type == 'text':
                                                text = part.get('text')
                                                print(f"Found text: {text}")
                                                if text:
                                                    text_parts.append(text)
                                elif content_type == 'user_editable_context':
                                    text = content.get('text')
                                    print(f"Found user context text: {text}")
                                    if text:
                                        text_parts.append(text)
                                
                                if text_parts:
                                    combined_text = ' '.join(text_parts)
                                    f.write(f"{role}: {combined_text}\n\n")
                    
        return output_file

    def _load_chat_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load chat data from conversations.json.
        
        Returns:
            Dictionary mapping chat IDs to lists of messages
        """
        chats = {}
        conversations_file = os.path.join(self.config.convo_folder, 'conversations.json')
        if not os.path.exists(conversations_file):
            print(f"No conversations.json found in {self.config.convo_folder}")
            return {}
            
        try:
            print(f"Loading conversations from {conversations_file}")
            with open(conversations_file, 'r') as f:
                conversations = json.load(f)
                
            if not isinstance(conversations, list):
                print(f"Error: Expected a list of conversations in {conversations_file}")
                return {}
                
            # Process each conversation
            for conv in conversations:
                if not isinstance(conv, dict):
                    continue
                    
                # Get required fields
                chat_id = conv.get('id')
                create_time = conv.get('create_time')
                if not chat_id or not create_time:
                    continue
                    
                # Filter by start date if specified
                if self.config.start_date:
                    create_date = datetime.fromtimestamp(create_time)
                    if create_date < self.config.start_date:
                        continue
                        
                # Extract messages
                messages = []
                for msg in conv.get('messages', []):
                    if not isinstance(msg, dict):
                        continue
                        
                    # Get message content
                    author = msg.get('author', {})
                    content = msg.get('content', {})
                    if not author or not content:
                        continue
                        
                    # Add to messages list
                    messages.append({
                        'author': author,
                        'content': content,
                        'create_time': msg.get('create_time', create_time)
                    })
                    
                # Sort messages by create time
                messages.sort(key=lambda x: x.get('create_time', 0))
                chats[chat_id] = messages
                
            print(f"\nLoaded {len(chats)} valid conversations")
            return chats
            
        except Exception as e:
            print(f"Error loading conversations: {str(e)}")
            return {}
