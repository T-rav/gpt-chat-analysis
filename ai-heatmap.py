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
        
        # Create OpenAI client
        self.openai_client = OpenAI(api_key=self.config.openai_api_key)
        
    def _load_conversations(self) -> List[Dict]:
        with open(f'{self.config.convo_folder}/conversations.json', 'r') as f:
            return json.load(f)
    
    def _save_single_chat(self, conv: Dict, research_folder: str) -> str:
        """Save a single chat conversation as markdown file and return the filepath."""
        chat_id = conv.get('id', '')
        title = conv.get('title', 'Untitled Chat')
        filename = f"{chat_id}.md"
        filepath = os.path.join(research_folder, filename)
        
        # Format the markdown content
        content = [f"# {title}\n"]
        content.append(f"Chat ID: {chat_id}\n")
        content.append(f"Created: {datetime.fromtimestamp(conv['create_time'], tz=timezone.utc)}\n")
        content.append("## Conversation\n")
        
        # Add messages
        for msg in conv.get('messages', []):
            role = msg.get('role', 'unknown')
            text = msg.get('content', '')
            content.append(f"### {role.capitalize()}:\n")
            content.append(f"{text}\n")
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        return filepath

    def save_chat_as_markdown(self, chat_id: str) -> None:
        """Save a specific chat conversation as a markdown file."""
        for conv in self.conversations:
            if conv.get('id') == chat_id:
                filepath = self._save_single_chat(conv, self.config.research_folder)
                print(f"Chat saved to {filepath}")
                return
        
        print(f"Chat with ID {chat_id} not found")

    def save_all_chats_parallel(self) -> None:
        """Save all chat conversations as markdown files using threading."""
        # Create research folder if it doesn't exist
        os.makedirs(self.config.research_folder, exist_ok=True)
        
        total_chats = len(self.conversations)
        print(f"Starting to save {total_chats} chats...\n")

        # Use ThreadPoolExecutor with progress bar
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for conv in self.conversations:
                future = executor.submit(self._save_single_chat, conv, self.config.research_folder)
                futures.append(future)
            
            # Show progress
            with tqdm(total=len(futures), desc="Saving chats") as pbar:
                results = []
                for future in futures:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

        print(f"\nSuccessfully saved {len(results)} chats to {self.config.research_folder}/")
        print("\nSaved files:")
        for filepath in sorted(results):
            print(f"- {os.path.basename(filepath)}")

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
    
    # Save all chats
    data.save_all_chats_parallel()

if __name__ == '__main__':
    main()
