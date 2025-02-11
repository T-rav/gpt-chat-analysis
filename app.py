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
                    {"role": "system", "content": """You are an expert analyst focused on evaluating how effectively users interact with AI systems. Analyze the USER's behavior in the following conversation. YOU MUST USE THE EXACT SECTION HEADINGS AND FORMAT PROVIDED BELOW:

# 1. Brief Summary
[Provide a concise overview of the USER's objectives and approach]

# 2. Five-Step Decision Loop Analysis

## Step 1: Problem Framing & Initial Prompting
- Effectiveness: [How well did the USER define and communicate their needs?]
- Evidence: [Specific examples of clear/unclear problem framing]
- Impact: [How this affected the conversation flow]

## Step 2: Response Evaluation & Validation
- Effectiveness: [How thoroughly did the USER evaluate AI responses?]
- Evidence: [Examples of verification, questioning, or acceptance]
- Impact: [How this shaped solution quality]

## Step 3: Expertise Application
- Effectiveness: [How well did the USER leverage their domain knowledge?]
- Evidence: [Examples of constraints or guidance provided]
- Impact: [How this improved solution relevance]

## Step 4: Critical Assessment
- Effectiveness: [How well did the USER assess limitations and risks?]
- Evidence: [Examples of testing assumptions or identifying issues]
- Impact: [How this prevented potential problems]

## Step 5: Process Improvement
- Effectiveness: [How did the USER refine and improve their approach?]
- Evidence: [Examples of learning and adaptation]
- Impact: [How this led to better outcomes]

## Overall Decision Loop Assessment
- Strongest Steps: [List the most effective steps]
- Areas for Enhancement: [Identify steps needing improvement]
- Emerging Patterns: [Note patterns across steps]

# 3. Collaborative Pattern Analysis

## Observed Patterns
- [List and analyze patterns the USER employed]
- [Evaluate effectiveness of collaboration]
- [Provide specific examples]

## Novel Patterns
- [Identify any unique approaches]
- [Assess their effectiveness]
- [Discuss potential benefits for others]

# 4. Recommendations
- [Specific suggestions for improvement]
- [Actionable steps for better AI collaboration]
- [Strategic adjustments to enhance outcomes]

You must maintain this exact structure and these exact headings in your response. Replace the text in brackets with your analysis while keeping the heading hierarchy and formatting consistent."""},
                    {"role": "user", "content": conversation}
                ],
                temperature=self.config.temperature
            )
            
            analysis = response.choices[0].message.content
            
            # Write just the analysis to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(analysis)
            
        except Exception as e:
            print(f"\nError analyzing chat {chat_id}: {str(e)}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Error: {str(e)}")
            return filepath, True  # Still processed, even though there was an error
        
        return filepath, True  # Successfully processed

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
        
        total_count = len(self.conversations)
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        print(f"\nStarting analysis of {total_count} conversations...")
        
        # Use ThreadPoolExecutor with progress bar
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for conv in self.conversations:
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
    
    def _process_timestamps(self) -> List[datetime]:
        """Process conversation timestamps into datetime objects."""
        return [
            datetime.fromtimestamp(conv['create_time'], tz=timezone.utc)
            .astimezone(pytz.timezone(self.config.local_tz))
            for conv in self.conversations
        ]
        




def main() -> None:
    """Main entry point for the chat analysis tool.
    
    This function:
    1. Parses command line arguments
    2. Sets up configuration
    3. Initializes the conversation processor
    4. Runs the analysis in parallel
    5. Optionally merges analysis into PDFs
    6. Handles any errors that occur
    """
    try:
        # Parse arguments
        args = CLIParser.parse_args()
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)
        
        # Initialize configuration
        config = Config(
            research_folder=args.output,
            pdf_chunks=args.pdf,
            pdf_output_dir=args.pdf_dir,
            pdf_size_limit_mb=args.pdf_size_limit
        )
        
        # Process conversations
        print(f"\nInitializing chat analysis...")
        data = ConversationData(config)
        
        # Run analysis
        print(f"\nStarting parallel analysis...")
        data.analyze_all_chats_parallel()
        
        # Generate PDFs if requested
        if args.pdf:
            print(f"\nMerging analysis into {args.pdf} PDF files...")
            pdf_gen = PDFGenerator(
                markdown_dir=args.output,
                output_dir=args.pdf_dir,
                size_limit_mb=config.pdf_size_limit_mb
            )
            pdf_gen.generate_pdfs(args.pdf)
        
        print(f"\nAnalysis complete! Results saved to: {args.output}")
        if args.pdf:
            print(f"PDF files saved to: {args.pdf_dir}")
        
    except ValueError as e:
        print(f"\nConfiguration Error: {str(e)}")
        exit(1)
    except FileNotFoundError as e:
        print(f"\nFile Error: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected Error: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()
