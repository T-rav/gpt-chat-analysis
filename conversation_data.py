"""Handles loading and processing of conversation data."""

import concurrent.futures
import inspect
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

    def analyze_single_chat(self, chat_id: str) -> None:
        """Analyze a single chat conversation.
        
        Args:
            chat_id: ID of the chat to analyze
        """
        # Load chat data
        chat_data = self._load_chat_data()
        
        if chat_id not in chat_data:
            raise ValueError(f"Chat ID {chat_id} not found in conversation data")
            
        # Create output directory
        os.makedirs(self.config.research_folder, exist_ok=True)
        
        # Analyze the single chat
        messages = chat_data[chat_id]
        filepath, status = self.analyze_and_save_chat(
            chat_id,
            messages,
            self.config.research_folder
        )
        print(f"Analysis of chat {chat_id} completed with status: {status}")
        print(f"Results saved to: {filepath}")

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
            return filepath, 'skipped'
            
        # Estimate token count (rough heuristic)
        estimated_tokens = sum(len(str(msg)) / 4 for msg in messages)
        if estimated_tokens > 120000:  # OpenAI's max context is 128k for 4o say under this
            print(f"Skipping chat {chat_id} - estimated {int(estimated_tokens)} tokens exceeds limit")
            return filepath, 'skipped'
        
        
        # Prepare conversation for analysis
        conversation = ""
        
        # Check if we're in single chat mode by looking at the call stack
        import inspect
        caller_frame = inspect.currentframe().f_back
        is_single_chat = caller_frame and caller_frame.f_code.co_name == 'analyze_single_chat'
        
        if is_single_chat:
            print("\nDebug - Processing chat messages:")
            print(f"Found {len(messages)} messages")
            print("Starting message processing...")
        
        for msg in messages:
            role = msg.get('author', {}).get('role', 'unknown')
            content = msg.get('content', {})
            
            # Get message text based on content type
            text = None
            content_type = content.get('content_type')
            parts = content.get('parts', [])
            
            if content_type == 'text':
                text = parts[0] if parts else None
            elif content_type == 'multimodal_text':
                # Combine all text parts
                text_parts = []
                for part in parts:
                    if isinstance(part, dict):
                        if part.get('content_type') in ['text', 'audio_transcription']:
                            text_parts.append(part.get('text', ''))
                text = ' '.join(text_parts) if text_parts else None
            elif content_type == 'user_editable_context':
                text = content.get('text')
            
            if text:
                if is_single_chat:
                    print(f"\nMessage:")
                    print(f"  Role: {role}")
                    print(f"  Type: {content_type}")
                    print(f"  Content: {text[:200]}..." if len(text) > 200 else f"  Content: {text}")
                conversation += f"{role}: {text}\n"
        
        # Print debug info before OpenAI call
        if is_single_chat:
            print("\nPreparing to call OpenAI API...")
            print(f"Total conversation length: {len(conversation)} characters")
            token_estimate = len(conversation.split()) * 1.3  # Rough estimate
            print(f"Estimated tokens: {int(token_estimate)}")
        
        # Analyze with OpenAI
        try:
            if is_single_chat:
                print("Calling OpenAI API...")
            
            # Set a timeout for the API call
            import httpx
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = self.openai_client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": self.config.system_prompt},
                            {"role": "user", "content": conversation}
                        ],
                        temperature=self.config.temperature
                    )
                
                if is_single_chat:
                    print("OpenAI API call completed successfully")
                
                # Save analysis to markdown file
                analysis = response.choices[0].message.content
                
            except httpx.TimeoutException:
                print(f"Error: API call timed out after 60 seconds")
                return filepath, 'api_error'
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                return filepath, 'cancelled'
            except Exception as e:
                print(f"Error during API call: {str(e)}")
                return filepath, 'api_error'
            
            # Only create directory and write file if we have valid analysis
            if analysis:
                os.makedirs(output_dir, exist_ok=True)
                
                # Write to a temporary file first
                temp_filepath = filepath + '.tmp'
                with open(temp_filepath, 'w') as f:
                    f.write(analysis)
                
                # Validate the format
                from file_validator import FileValidator
                if FileValidator.verify_md_format(temp_filepath):
                    # If valid, rename to final filepath
                    os.rename(temp_filepath, filepath)
                    return filepath, 'success'
                else:
                    # If invalid and in single chat mode, print the analysis content
                    if is_single_chat:
                        print("\nAnalysis content that failed format validation:")
                        print("=" * 80)
                        print(analysis)
                        print("=" * 80)
                        print("\nMissing required sections:")
                        FileValidator.verify_md_format(temp_filepath, debug=True)
                    
                    # Delete temp file and count as format error
                    os.remove(temp_filepath)
                    print(f"Format error in chat {chat_id} - generated analysis has invalid format")
                    return filepath, 'format_error'
            
            return filepath, 'api_error'
            
        except Exception as e:
            print(f"Error analyzing chat {chat_id}: {str(e)}")
            # Don't create the file if analysis failed
            return os.path.join(output_dir, f"{chat_id}.md"), 'api_error'

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
            skipped = 0
            format_errors = 0
            api_errors = 0
            
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                _, status = future.result()
                
                if status == 'success':
                    successful += 1
                elif status == 'skipped':
                    skipped += 1
                elif status == 'format_error':
                    format_errors += 1
                elif status == 'api_error':
                    api_errors += 1
                    
                print(f"\rProgress: {completed}/{len(futures)} chats processed", end='')
                
            print(f"\nCompleted! {successful}/{len(futures)} chats analyzed successfully")
            print(f"Skipped: {skipped} chats (already exist or too large)")
            print(f"Failed due to format errors: {format_errors} chats")
            print(f"Failed due to API errors: {api_errors} chats")
            print(f"Total failed: {format_errors + api_errors} chats")

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
            format: Output format ('json' or 'txt'). Any non-JSON format will be exported as TXT.
            
        Returns:
            Path to the exported file
        """
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
                    print(f"Skipping non-dict conversation: {type(conv)}")
                    continue
                    
                # Get required fields
                chat_id = conv.get('id')
                create_time = conv.get('create_time')
                if not chat_id or not create_time:
                    print(f"Skipping conversation missing id or create_time: {chat_id}")
                    continue
                    
                # Debug: Print more info if this is the chat we're looking for
                caller_frame = inspect.currentframe().f_back
                is_single_chat = caller_frame and caller_frame.f_code.co_name == 'analyze_single_chat'
                if is_single_chat and chat_id == caller_frame.f_locals.get('chat_id'):
                    print(f"\nFound target chat {chat_id}:")
                    print(f"  create_time: {create_time}")
                    print(f"  messages: {len(conv.get('messages', []))}")
                    print(f"  mapping: {len(conv.get('mapping', {}))}")
                    print(f"  Full conversation data: {json.dumps(conv, indent=2)}\n")
                    
                # Filter by start date if specified
                if self.config.start_date:
                    create_date = datetime.fromtimestamp(create_time)
                    if create_date < self.config.start_date:
                        continue
                        
                # Extract messages from mapping
                messages = []
                mapping = conv.get('mapping', {})
                current_node = conv.get('current_node')
                
                # Debug: Print message counts if in single chat mode
                if is_single_chat:
                    print(f"Processing messages from mapping:")
                    print(f"Total messages in mapping: {len(mapping)}")
                    print(f"Current node: {current_node}")
                
                # Function to traverse message tree
                def traverse_messages(node_id, visited=None):
                    if visited is None:
                        visited = set()
                    
                    if node_id in visited:
                        return []
                    visited.add(node_id)
                    
                    node_data = mapping.get(node_id)
                    if not node_data:
                        return []
                    
                    # Get message data
                    message = node_data.get('message')
                    result = []
                    
                    # Add parent's message first
                    parent_id = node_data.get('parent')
                    if parent_id:
                        result.extend(traverse_messages(parent_id, visited))
                    
                    # Add current message if valid
                    if message:
                        author = message.get('author', {})
                        content = message.get('content', {})
                        metadata = message.get('metadata', {})
                        
                        # Skip system messages and hidden messages
                        if (author.get('role') != 'system' and 
                            not metadata.get('is_visually_hidden_from_conversation', False)):
                            
                            parts = content.get('parts', [])
                            if parts and parts[0]:
                                if is_single_chat:
                                    print(f"\nIncluding message {node_id}:")
                                    print(f"  role: {author.get('role')}")
                                    print(f"  content: {parts[0][:100]}..." if len(parts[0]) > 100 else f"  content: {parts[0]}")
                                result.append({
                                    'author': author,
                                    'content': content,
                                    'create_time': message.get('create_time', create_time)
                                })
                            elif is_single_chat:
                                print(f"Skipping message {node_id} - empty content")
                        elif is_single_chat:
                            print(f"Skipping message {node_id} - system or hidden message")
                    
                    return result
                
                # Start traversal from current node
                if current_node:
                    messages = traverse_messages(current_node)
                    if is_single_chat:
                        print(f"\nFound {len(messages)} messages in conversation")
                
                chats[chat_id] = messages
                
            print(f"\nLoaded {len(chats)} valid conversations")
            return chats
            
        except Exception as e:
            print(f"Error loading conversations: {str(e)}")
            return {}
