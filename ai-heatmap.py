from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any
import json
import pytz
import os
import re
from multiprocessing import cpu_count
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Configuration
CONVO_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chats')
LOCAL_TZ = 'US/Mountain'

@dataclass
class Config:
    convo_folder: str = CONVO_FOLDER
    local_tz: str = LOCAL_TZ
    research_folder: str = 'analysis'
    openai_api_key: str = None
    model: str = 'gpt-4o'
    temperature: float = 0.62

    def __post_init__(self):
        # Load environment variables
        load_dotenv()
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')

class ConversationData:
    def __init__(self, config: Config):
        self.config = config
        self.conversations = self._load_conversations()
        self.convo_times = self._process_timestamps()
        
        # Validate OpenAI API key
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key is required for analysis. Please set it in your .env file.")
        
        # Create OpenAI client with base configuration
        self.openai_client = OpenAI(api_key=self.config.openai_api_key)
        
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
    
    def analyze_and_save_chat(self, conv: Dict, research_folder: str) -> str:
        """Analyze a chat with OpenAI and save just the analysis to markdown."""
        chat_id = conv.get('id', '')
        filename = f"{chat_id}.md"
        filepath = os.path.join(research_folder, filename)
        
        # Extract messages from mapping
        mapping = conv.get('mapping', {})
        if not mapping:
            print(f"Warning: No mapping found in chat {chat_id}")
            return filepath
            
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
            return filepath
        
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
                    {"role": "system", "content": """You are an expert conversation analyst focused on evaluating AI interactions against two frameworks while also discovering new patterns:

I. The 5-step AI Decision Loop:
1. Frame the Decision Context & Guide AI Prompting
   - Problem/goal definition and constraints
   - Clear AI role and context specification
   - Structured prompting with reasoning requests

2. Generate Multi-Perspective AI Outputs & Validate
   - Multiple perspectives/alternatives requested
   - Validation for accuracy and coherence
   - Self-critiquing and external reference checks

3. Apply Human Judgment & Adjust AI Interaction
   - Human expertise integration
   - Real-world constraint application
   - Iterative refinement of AI outputs

4. Test for Bias & Feasibility
   - What-if scenarios and outcome simulation
   - Bias identification and mitigation
   - Implementation feasibility assessment

5. Refine, Iterate, and Automate
   - Feedback collection and metric tracking
   - Process improvement and automation
   - Documentation and knowledge base building

II. Known Collaborative Work Patterns:
1. Iterative Refinement Pattern
   - Human proposes → AI refines
   - AI drafts → Human iterates
   - Human outlines → AI expands

2. Review and Adjustment Pattern
   - AI self-critiques when asked
   - Human requests refinements
   - Mutual quality checks

3. Reasoning and Challenge Pattern
   - AI provides alternatives/counterarguments
   - Explicit thought process sharing
   - 'Why' questions for clarity

For the given conversation, analyze and provide:
1. Brief summary of the interaction

2. Analysis of the AI Decision Loop:
   - Which steps were present and effective?
   - What was missing or could be improved?
   - Specific examples from the conversation

3. Analysis of Collaborative Patterns:
   - Which known patterns were demonstrated?
   - How effective was the collaboration?
   - Examples of successful exchanges
   - NEW PATTERNS DISCOVERED: Identify and describe any novel collaboration patterns
     * What unique interaction patterns emerged?
     * How did they enhance the collaboration?
     * Could these patterns be formalized for future use?

4. Recommendations for improvement:
   - Decision loop enhancements
   - Collaboration pattern suggestions (both known and new)
   - Specific prompting strategies

5. Notable insights about:
   - Decision-making process
   - Collaboration effectiveness
   - Pattern evolution and innovation
   - Learning opportunities"""},
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
        
        total_chats = len(self.conversations)
        print(f"Starting to analyze {total_chats} chats...\n")

        # Use ThreadPoolExecutor with progress bar
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for conv in self.conversations:
                future = executor.submit(self.analyze_and_save_chat, conv, self.config.research_folder)
                futures.append(future)
            
            # Show progress
            with tqdm(total=len(futures), desc="Analyzing chats") as pbar:
                results = []
                for future in futures:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

        print(f"\nSuccessfully analyzed {len(results)} chats to {self.config.research_folder}/")
        print("\nAnalysis files:")
        for filepath in sorted(results):
            print(f"- {os.path.basename(filepath)}")
            
    def _process_timestamps(self) -> List[datetime]:
        """Process conversation timestamps into datetime objects."""
        return [
            datetime.fromtimestamp(conv['create_time'], tz=timezone.utc)
            .astimezone(pytz.timezone(self.config.local_tz))
            for conv in self.conversations
        ]
        


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process and analyze chat conversations')
    parser.add_argument('-o', '--output', type=str, default='analysis',
                      help='Output directory for analysis files')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create config with output directory
    config = Config(research_folder=args.output)
    
    # Initialize conversation data
    data = ConversationData(config)
    
    # Analyze all chats and save analysis
    data.analyze_all_chats_parallel()

if __name__ == '__main__':
    main()
