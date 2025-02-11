"""Configuration module for the AI heatmap analysis."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

@dataclass
class Config:
    """Configuration for the conversation analysis.
    
    Attributes:
        convo_folder: Directory containing conversation files (default: './chats')
        research_folder: Output directory for analysis files (default: 'analysis')
        local_tz: Local timezone for timestamp processing (default: 'US/Mountain')
        openai_api_key: API key for OpenAI services (from env or passed directly)
        model: GPT model to use for analysis (default: 'gpt-4')
        temperature: Temperature setting for GPT responses (default: 0.2)
        max_workers: Maximum number of parallel workers (default: min(8, CPU_COUNT))
    """
    # Paths (with defaults)
    convo_folder: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chats')
    research_folder: str = 'analysis'
    
    # Timezone settings
    local_tz: str = 'US/Mountain'
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    model: str = 'gpt-4o'
    temperature: float = 0.2
    
    # Processing settings
    max_workers: int = min(8, os.cpu_count() or 4)  # Limit to 8 or CPU count, whichever is smaller

    def __post_init__(self) -> None:
        """Initialize configuration with environment variables."""
        load_dotenv()
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required. Set it in your .env file or pass it to Config.")
        
        # Ensure paths exist
        os.makedirs(self.convo_folder, exist_ok=True)
        os.makedirs(self.research_folder, exist_ok=True)
