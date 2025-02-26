import os
from openai import OpenAI
from typing import Dict, Any
from configuration import Config
class TrendProcessor:
    """Handles analysis of markdown files for chat completion statistics.
    
    This class is responsible for:
    1. Reading markdown files from a specified directory
    2. Analyzing loop completion patterns
    3. Generating statistical summaries
    """
    
    def __init__(self):
        """Initialize the analysis processor with OpenAI client."""
        config = Config()
        if not config.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.model
        self.temperature = config.temperature
    
    def analyze_directory(self, directory):
        """Analyze markdown files in the specified directory for loop completion.
        
        Args:
            directory (str): Path to directory containing markdown files
            
        Returns:
            dict: Analysis summary with completion statistics
            
        Raises:
            FileNotFoundError: If the specified directory doesn't exist
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
            
        total_chats = 0
        completed_loops = 0
        not_completed_loops = 0
        
        # Process each markdown file
        print("\nAnalyzing files:")
        for filename in os.listdir(directory):
            if filename.endswith(".md"):
                filepath = os.path.join(directory, filename)
                stats = self._process_file(filepath)
                total_chats += stats['total']
                completed_loops += stats['completed']
                not_completed_loops += stats['not_completed']
                
                # Print per-file results
                status = "✓" if stats['completed'] == 1 else "✗"
                print(f"{status} {filename}")
        
        return self._generate_summary(total_chats, completed_loops, not_completed_loops)
    
    def _analyze_with_openai(self, text: str) -> Dict[str, Any]:
        """Analyze text using OpenAI to determine loop completion.
        
        Args:
            text (str): The conversation text to analyze
            
        Returns:
            dict: Analysis results indicating if loop was completed
        """
        prompt = (
            "Analyze the following conversation and determine if the user completed all five steps "
            "of the AI Decision Loop. Look for evidence of: 1) Problem Framing, 2) Solution Design, "
            "3) Implementation, 4) Testing & Validation, and 5) Iteration & Refinement.\n\n"
            "Respond with only 'yes' or 'no'.\n\nConversation:\n"
            f"{text}"
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        
        result = response.choices[0].message.content.strip().lower()
        return {'completed': result == 'yes', 'not_completed': result == 'no'}

    def _process_file(self, file_path: str) -> Dict[str, int]:
        """Process a single markdown file and extract completion statistics.
        
        Args:
            file_path (str): Path to the markdown file
            
        Returns:
            dict: Statistics for the file including total, completed, and not completed loops
        """
        with open(file_path, "r", encoding="utf-8") as file:
            md_text = file.read()
        
        analysis = self._analyze_with_openai(md_text)
        
        return {
            'total': 1,  # Each file represents one conversation
            'completed': 1 if analysis['completed'] else 0,
            'not_completed': 1 if analysis['not_completed'] else 0
        }
    
    def _generate_summary(self, total_chats, completed_loops, not_completed_loops):
        """Generate a summary of the analysis results.
        
        Args:
            total_chats (int): Total number of chats analyzed
            completed_loops (int): Number of completed loops
            not_completed_loops (int): Number of not completed loops
            
        Returns:
            dict: Summary statistics including percentages
        """
        completed_percentage = (completed_loops / total_chats) * 100 if total_chats > 0 else 0
        not_completed_percentage = (not_completed_loops / total_chats) * 100 if total_chats > 0 else 0
        
        return {
            "Total Chats Analyzed": total_chats,
            "Completed Loop (%)": completed_percentage,
            "Not Completed Loop (%)": not_completed_percentage,
        }
