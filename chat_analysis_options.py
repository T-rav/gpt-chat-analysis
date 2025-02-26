import os
from typing import Dict, Any

from cli import CLIParser
from configuration import Config
from conversation_data import ConversationData
from pdf_generator import PDFGenerator
from trend_processor import TrendProcessor

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

    def analyze_trends(self) -> None:
        """Analyze trends in markdown files from the specified directory or single chat."""
        try:
            # Use the output directory from config for JSON analysis files
            analyzer = TrendProcessor(output_dir=self.config.research_folder)
            
            # Determine the analysis directory (either from --trends or -o)
            analysis_dir = self.args.trends if self.args.trends else self.args.output
            
            if self.args.chat_id:
                # Single chat analysis
                target_file = os.path.join(analysis_dir, f"{self.args.chat_id}.md")
                if not os.path.isfile(target_file):
                    raise FileNotFoundError(f"Chat file not found: {target_file}")
                    
                print(f"\nAnalyzing single chat: {self.args.chat_id}")
                stats = analyzer._process_file(target_file)
                print("\nAnalysis Results:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                    
            elif analysis_dir:
                # Full directory analysis
                print(f"\nAnalyzing trends in: {analysis_dir}")
                summary = analyzer.analyze_directory(analysis_dir)
                
                print("\nTrends Summary:")
                for section, data in summary.items():
                    print(f"\n{section}:")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, float):
                                print(f"  {key}: {value:.2f}")
                            else:
                                print(f"  {key}: {value}")
                    else:
                        print(f"  {data}")
            else:
                print("Error: Must specify either --trends or -o for analysis directory")
        except Exception as e:
            print(f"Error analyzing trends: {str(e)}")
    
    def analyze_chats(self) -> None:
        """Run analysis on chat conversations."""
        print(f"\nInitializing chat analysis...")
        data = ConversationData(self.config)
        
        if self.args.chat_id:
            print(f"\nAnalyzing single chat: {self.args.chat_id}")
            data.analyze_single_chat(self.args.chat_id)
        else:
            print(f"\nStarting parallel analysis of all chats...")
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
            elif self.args.trends:
                self.analyze_trends()
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
