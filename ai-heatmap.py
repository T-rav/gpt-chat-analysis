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
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Union, Any

# Third-party imports
import pytz
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Local imports
from config import (
    CONVO_FOLDER,
    LOCAL_TZ,
    RESEARCH_FOLDER,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_WORKERS
)

@dataclass
class Config:
    """Configuration for the conversation analysis.
    
    Attributes:
        convo_folder: Directory containing conversation files
        local_tz: Local timezone for timestamp processing
        research_folder: Output directory for analysis files
        openai_api_key: API key for OpenAI services
        model: GPT model to use for analysis
        temperature: Temperature setting for GPT responses
    """
    convo_folder: str = CONVO_FOLDER
    local_tz: str = LOCAL_TZ
    research_folder: str = RESEARCH_FOLDER
    openai_api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE

    def __post_init__(self) -> None:
        """Initialize configuration with environment variables."""
        load_dotenv()
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required. Set it in your .env file or pass it to Config.")

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
                    # Print structure of first conversation
                    if data:
                        print(f"\nFirst conversation structure:")
                        first_conv = data[0]
                        print(f"Type: {type(first_conv)}")
                        if isinstance(first_conv, dict):
                            print(f"Keys: {list(first_conv.keys())}")
                    
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
                    {"role": "system", "content": """You are an expert analyst focused on evaluating how effectively users interact with AI systems. Analyze the USER's behavior in the following conversation:

I. User's Decision-Making Process:
1. Problem Framing & Initial Prompting
   - How clearly did the USER define their needs?
   - Did they provide necessary context?
   - How structured were their requests?

2. Response Evaluation & Validation
   - Did the USER verify or question outputs?
   - Did they ask for alternatives?
   - How did they validate quality?

3. Expertise Application
   - How did the USER apply their knowledge?
   - Did they provide domain constraints?
   - How did they guide refinements?

4. Critical Assessment
   - Did the USER consider limitations?
   - How did they handle potential issues?
   - Did they test assumptions?

5. Process Improvement
   - Did the USER refine their approach?
   - How did they handle iterations?
   - Did they document learnings?

II. Collaborative Work Patterns (Both USER and Assistant):
1. Iterative Refinement
   - USER proposes → Assistant refines
   - Assistant drafts → USER iterates
   - USER outlines → Assistant expands

2. Review and Adjustment
   - USER requests improvements
   - Quality check exchanges
   - Refinement cycles

3. Reasoning and Challenge
   - USER asks 'why' questions
   - Thought process sharing
   - Alternative explorations

For the given conversation, analyze and provide:
1. Brief summary focusing on the USER's objectives and approach

2. Analysis of USER's Decision-Making:
   - Strengths in their approach
   - Areas for improvement
   - Specific examples of effective/ineffective interactions

3. Collaborative Pattern Analysis:
   - Which patterns did the USER employ?
   - How effectively did they guide the collaboration?
   - Examples of successful exchanges
   - NEW PATTERNS: Identify any novel ways the USER worked with the AI
     * What unique approaches did they take?
     * How effective were these approaches?
     * Could others benefit from these patterns?

4. Recommendations for the USER:
   - How to frame requests better
   - Ways to guide the interaction more effectively
   - Specific prompting strategies

5. Key Insights about:
   - USER's interaction style
   - Effective techniques they employed
   - Learning opportunities for better AI collaboration

6. Decision Loop Model Evolution:
   - NEW STEPS DISCOVERED: Did this conversation reveal any missing steps in our decision loop model?
   - MISSING ELEMENTS: Were there important aspects of user decision-making that don't fit in any current step?
   - MODEL IMPROVEMENTS: How could the decision loop model be enhanced based on this conversation?
   Examples:
   - A new step between existing steps
   - A missing aspect within a step
   - A different way to structure the steps
   - Additional dimensions to consider"""},
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
        
        return filepath

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
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
        


def parse_args() -> Any:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Analyze chat conversations using AI to evaluate decision-making and collaboration patterns'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='analysis',
        help='Output directory for analysis files'
    )
    return parser.parse_args()

def main() -> None:
    """Main entry point for the chat analysis tool.
    
    This function:
    1. Parses command line arguments
    2. Sets up configuration
    3. Initializes the conversation processor
    4. Runs the analysis in parallel
    5. Handles any errors that occur
    """
    try:
        args = parse_args()
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)
        
        # Initialize configuration
        config = Config(research_folder=args.output)
        
        # Process conversations
        print(f"\nInitializing chat analysis...")
        data = ConversationData(config)
        
        # Run analysis
        print(f"\nStarting parallel analysis...")
        data.analyze_all_chats_parallel()
        
        print(f"\nAnalysis complete! Results saved to: {args.output}")
        
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
