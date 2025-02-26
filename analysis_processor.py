"""Module for analyzing markdown files containing chat completion data."""

import os
import re

# todo : use LLM and aggregate stats
class AnalysisProcessor:
    """Handles analysis of markdown files for chat completion statistics.
    
    This class is responsible for:
    1. Reading markdown files from a specified directory
    2. Analyzing loop completion patterns
    3. Generating statistical summaries
    """
    
    def __init__(self):
        """Initialize the analysis processor with pattern matching rules."""
        self.completion_pattern = re.compile(
            r"### 4\.1 Loop Completion Analysis\n- \*\*Did the USER complete all five steps of the AI Decision Loop\?\*\*\n\s*-\s*(Yes|No)", 
            re.IGNORECASE
        )
    
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
        for filename in os.listdir(directory):
            if filename.endswith(".md"):
                stats = self._process_file(os.path.join(directory, filename))
                total_chats += stats['total']
                completed_loops += stats['completed']
                not_completed_loops += stats['not_completed']
        
        return self._generate_summary(total_chats, completed_loops, not_completed_loops)
    
    def _process_file(self, file_path):
        """Process a single markdown file and extract completion statistics.
        
        Args:
            file_path (str): Path to the markdown file
            
        Returns:
            dict: Statistics for the file including total, completed, and not completed loops
        """
        with open(file_path, "r", encoding="utf-8") as file:
            md_text = file.read()
            
        loop_completion_sections = self.completion_pattern.findall(md_text)
        
        return {
            'total': len(loop_completion_sections),
            'completed': sum(1 for result in loop_completion_sections if result.lower() == "yes"),
            'not_completed': sum(1 for result in loop_completion_sections if result.lower() == "no")
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
