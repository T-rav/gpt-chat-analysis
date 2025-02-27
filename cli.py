"""Command-line interface for the chat analysis tool."""

import argparse
from datetime import datetime
from typing import Any

class CLIParser:
    """Handles command-line argument parsing for the chat analysis tool."""
    
    @staticmethod
    def _validate_positive_int(value: str, param: str) -> int:
        """Validate that the value is a positive integer.
        
        Args:
            value: The value to validate
            param: The parameter name for error messages
            
        Returns:
            int: The validated positive integer
            
        Raises:
            argparse.ArgumentTypeError: If validation fails
        """
        try:
            ivalue = int(value)
            if ivalue <= 0:
                raise argparse.ArgumentTypeError(
                    f"{param} must be a positive integer, got {value}"
                )
            return ivalue
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{param} must be a positive integer, got {value}"
            )
    
    @staticmethod
    def _validate_positive_float(value: str, param: str) -> float:
        """Validate that the value is a positive float.
        
        Args:
            value: The value to validate
            param: The parameter name for error messages
            
        Returns:
            float: The validated positive float
            
        Raises:
            argparse.ArgumentTypeError: If validation fails
        """
        try:
            fvalue = float(value)
            if fvalue <= 0:
                raise argparse.ArgumentTypeError(
                    f"{param} must be a positive number, got {value}"
                )
            return fvalue
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{param} must be a positive number, got {value}"
            )
    
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
            type=lambda x: CLIParser._validate_positive_int(x, '--pdf'),
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
            type=lambda x: CLIParser._validate_positive_float(x, '--pdf-size-limit'),
            default=1.0,
            help='Maximum size in MB for each PDF file (default: 1MB)'
        )
        parser.add_argument(
            '-d', '--date',
            type=lambda d: datetime.strptime(d, '%Y-%m-%d').date(),
            help='Start date for analysis (format: YYYY-MM-DD)'
        )
        parser.add_argument(
            '--export-chat',
            type=str,
            help='Export chat history for a specific conversation ID to a file'
        )
        parser.add_argument(
            '--export-format',
            type=str,
            choices=['json', 'txt'],
            default='txt',
            help='Format for chat export (default: txt)'
        )
        parser.add_argument(
            '--trends',
            type=str,
            help='Analyze trends in markdown files from the specified directory'
        )
        parser.add_argument(
            '--verify-format',
            action='store_true',
            help='Verify markdown files format and remove invalid files (uses directory specified by --output)'
        )
        parser.add_argument(
            '--chat-id',
            type=str,
            help='Process a single chat ID for analysis'
        )
        parser.add_argument(
            '--force-reprocess',
            action='store_true',
            help='Force reprocessing of all files, ignoring cached results'
        )
        return parser.parse_args()
