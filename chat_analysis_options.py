import os
from typing import Dict, Any

from cli import CLIParser
from configuration import Config
from conversation_data import ConversationData
from pdf_generator import PDFGenerator
from analysis_processor import AnalysisProcessor

class ChatAnalysisOptions:
    """Main application class for chat analysis.
    
    This class handles:
    1. Command-line argument processing
    2. Configuration management
    3. Coordinating different analysis modes (chat export, PDF generation, markdown analysis)
    4. Error handling and reporting
    """
    
    def __init__(self):
        """Initialize the chat analysis application."""
        self.args = CLIParser.parse_args()
        self.config = self._initialize_config()
    
    def _initialize_config(self) -> Config:
        """Initialize configuration based on command line arguments.
        
        Returns:
            Config: Application configuration
        """
        os.makedirs(self.args.output, exist_ok=True)
        return Config(
            research_folder=self.args.output,
            pdf_chunks=self.args.pdf,
            pdf_output_dir=self.args.pdf_dir,
            pdf_size_limit_mb=self.args.pdf_size_limit,
            start_date=self.args.date
        )
    
    def export_chat(self) -> None:
        """Export a specific chat conversation."""
        print(f"\nExporting chat {self.args.export_chat}...")
        data = ConversationData(self.config)
        output_file = data.export_chat_history(self.args.export_chat, self.args.export_format)
        print(f"Chat exported to: {output_file}")
    
    def generate_pdfs(self) -> None:
        """Generate PDF files from markdown analysis."""
        print(f"\nGenerating {self.args.pdf} PDF files from existing markdown...")
        pdf_gen = PDFGenerator(
            markdown_dir=self.args.output,
            output_dir=self.args.pdf_dir,
            size_limit_mb=self.config.pdf_size_limit_mb
        )
        pdf_gen.generate_pdfs(self.args.pdf)
    
    def verify_markdown_format(self) -> None:
        """Verify and clean markdown files in the analysis directory."""
        from file_validator import FileValidator
        directory = self.args.output
        print(f"\nVerifying markdown files in: {directory}")
        try:
            invalid_files, total_invalid = FileValidator.verify_and_clean_md_files(directory)
            if invalid_files:
                print(f"\nFound {total_invalid} invalid files:")
                for file in invalid_files:
                    print(f"  - {file}")
                print(f"All invalid files have been removed from {directory}")
            else:
                print("All markdown files have valid format")
        except Exception as e:
            print(f"Error during verification: {str(e)}")

    def analyze_markdown_files(self) -> None:
        """Analyze markdown files in the specified directory."""
        print(f"\nAnalyzing markdown files in: {self.args.analyze}")
        try:
            analyzer = AnalysisProcessor()
            summary = analyzer.analyze_directory(self.args.analyze)
            print("\nAnalysis Summary:")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
    
    def analyze_chats(self) -> None:
        """Run parallel analysis on all chat conversations."""
        print(f"\nInitializing chat analysis...")
        data = ConversationData(self.config)
        print(f"\nStarting parallel analysis...")
        data.analyze_all_chats_parallel()
        
        print(f"\nAnalysis complete! Results saved to: {self.args.output}")
        if self.args.pdf:
            print(f"PDF files saved to: {self.args.pdf_dir}")
    
    def run(self) -> None:
        """Run the chat analysis application."""
        try:
            if self.args.verify_format:
                self.verify_markdown_format()
            if self.args.export_chat:
                self.export_chat()
            elif self.args.pdf:
                self.generate_pdfs()
            elif self.args.analyze:
                self.analyze_markdown_files()
            elif not self.args.verify_format:  # Only analyze chats if not just verifying format
                self.analyze_chats()
                
        except ValueError as e:
            print(f"\nConfiguration Error: {str(e)}")
            exit(1)
        except FileNotFoundError as e:
            print(f"\nFile Error: {str(e)}")
            exit(1)
        except Exception as e:
            print(f"\nUnexpected Error: {str(e)}")
            exit(1)
