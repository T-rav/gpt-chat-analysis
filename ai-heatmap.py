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
        self.openai_client = OpenAI()
        
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
        
        # Add AI Analysis if available
        if hasattr(self, 'analysis_results') and chat_id in self.analysis_results:
            content.append("## AI Analysis\n")
            content.append(f"{self.analysis_results[chat_id]}\n")
        
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
            
    def _process_timestamps(self) -> List[datetime]:
        """Process conversation timestamps into datetime objects."""
        return [
            datetime.fromtimestamp(conv['create_time'], tz=timezone.utc)
            .astimezone(pytz.timezone(self.config.local_tz))
            for conv in self.conversations
        ]
        
    def analyze_multiple_chats(self, chat_ids: List[str] = None) -> Dict[str, str]:
        """Analyze multiple chat conversations using OpenAI API and save results."""
        if chat_ids is None:
            # If no specific chat IDs provided, analyze all conversations
            chat_ids = [conv['id'] for conv in self.conversations]
        
        print(f"\nAnalyzing {len(chat_ids)} chats...")
        
        self.analysis_results = {}
        with tqdm(total=len(chat_ids), desc="Analyzing chats") as pbar:
            for chat_id in chat_ids:
                # Find the conversation
                conv = next((c for c in self.conversations if c['id'] == chat_id), None)
                if not conv:
                    print(f"Warning: Chat {chat_id} not found")
                    continue
                
                # Extract messages
                messages = conv.get('messages', [])
                if not messages:
                    print(f"Warning: No messages found in chat {chat_id}")
                    continue
                
                # Prepare conversation for analysis
                conversation = ""
                for msg in messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    conversation += f"{role}: {content}\n"
                
                # Analyze with OpenAI
                try:
                    response = self.openai_client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": "You are an expert conversation analyst. Analyze the following chat conversation and provide a concise summary of the key points, topics discussed, and any notable patterns or insights."},
                            {"role": "user", "content": conversation}
                        ],
                        temperature=self.config.temperature
                    )
                    
                    analysis = response.choices[0].message.content
                    self.analysis_results[chat_id] = analysis
                    
                except Exception as e:
                    print(f"\nError analyzing chat {chat_id}: {str(e)}")
                    self.analysis_results[chat_id] = f"Error: {str(e)}"
                
                pbar.update(1)
        
        return self.analysis_results

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
    
    # First analyze all chats
    print("Starting chat analysis...")
    analysis_results = data.analyze_multiple_chats()
    
    # Then save all chats with their analysis
    print("\nSaving chats with analysis...")
    data.save_all_chats_parallel()

if __name__ == '__main__':
    main()
