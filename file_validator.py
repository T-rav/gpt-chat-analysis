"""Module for validating and cleaning markdown files."""

import os
import logging
from typing import List, Tuple

class FileValidator:
    """Handles validation and cleaning of markdown files."""
    
    @staticmethod
    def verify_md_format(file_path: str, debug: bool = False) -> bool:
        """Verify if a markdown file has the correct format.
        
        Args:
            file_path: Path to the markdown file
            debug: If True, print detailed validation info
            
        Returns:
            bool: True if format is valid, False otherwise
        """
        required_sections = [
            '# 1. Brief Summary',
            '# 2. Five-Step Decision Loop Analysis',
            '## Step 1: Problem Framing & Initial Prompting',
            '## Step 2: Response Evaluation & Validation',
            '## Step 3: Expertise Application',
            '## Step 4: Critical Assessment',
            '### 4.1 Loop Completion Analysis',
            '### 4.2 Breakdown Analysis',
            '## Step 5: Process Improvement',
            '# 3. Collaborative Pattern Analysis',
            '## Observed Patterns',
            '## Novel Patterns',
            '# 4. Recommendations'
        ]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for generic content that indicates an incomplete analysis
            generic_content_indicators = [
                "The USER engaged with the AI",
                "objectives and approach are not provided",
                "lack of conversation content"
            ]
            
            for indicator in generic_content_indicators:
                if indicator in content:
                    if debug:
                        print(f"Found generic content indicator: '{indicator}'")
                    logging.warning(f"File {file_path} contains generic placeholder content")
                    return False
                
            # Check if all required sections are present
            missing_sections = []
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                if debug:
                    print("Missing sections:")
                    for section in missing_sections:
                        print(f"  - {section}")
                logging.warning(f"File {file_path} is missing required sections: {', '.join(missing_sections)}")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {str(e)}")
            return False

    @staticmethod
    def verify_and_clean_md_files(directory: str) -> Tuple[List[str], int]:
        """Verify all markdown files in directory and remove invalid ones.
        
        Args:
            directory: Directory containing markdown files
            
        Returns:
            Tuple containing list of invalid files and total count
        """
        invalid_files = []
        
        if not os.path.exists(directory):
            logging.error(f"Directory {directory} does not exist")
            return [], 0
        
        for filename in os.listdir(directory):
            if filename.endswith('.md'):
                file_path = os.path.join(directory, filename)
                if not FileValidator.verify_md_format(file_path):
                    invalid_files.append(filename)
                    try:
                        os.remove(file_path)
                        logging.info(f"Removed invalid file: {filename}")
                    except Exception as e:
                        logging.error(f"Error removing file {filename}: {str(e)}")
        
        return invalid_files, len(invalid_files)
