"""Command-line interface for the chat analysis tool."""

import argparse
from datetime import datetime
from typing import Any

class CLIParser:
    """Handles command-line argument parsing for the chat analysis tool."""
    
    @staticmethod
    def parse_args() -> Any:
        """Parse command line arguments.
        
        Returns:
            argparse.Namespace: Parsed command line arguments
        """
        parser = argparse.ArgumentParser(
            description='Analyze chat conversations using AI to evaluate decision-making and collaboration patterns'
        )
        parser.add_argument(
            '-o', '--output',
            type=str,
            default='analysis',
            help='Output directory for analysis files'
        )
        parser.add_argument(
            '--pdf',
            type=int,
            help='Merge analysis into specified number of PDF files'
        )
        parser.add_argument(
            '--pdf-dir',
            type=str,
            default='pdf_analysis',
            help='Output directory for PDF files'
        )
        parser.add_argument(
            '--pdf-size-limit',
            type=float,
            default=1.0,
            help='Maximum size in MB for each PDF file (default: 1MB)'
        )
        parser.add_argument(
            '-d', '--date',
            type=lambda d: datetime.strptime(d, '%Y-%m-%d').date(),
            help='Start date for analysis (format: YYYY-MM-DD)'
        )
        return parser.parse_args()
