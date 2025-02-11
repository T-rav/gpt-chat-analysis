"""PDF generation and merging functionality."""

import tempfile
from pathlib import Path
from typing import List, Tuple

import markdown2
from PyPDF2 import PdfMerger
from tqdm import tqdm
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

class PDFGenerator:
    """Handles conversion of markdown files to PDFs and merging of PDFs."""
    
    def __init__(self, markdown_dir: str, output_dir: str, size_limit_mb: float = 10.0):
        """Initialize the PDF generator.
        
        Args:
            markdown_dir: Directory containing markdown files
            output_dir: Directory to save PDF files
            size_limit_mb: Maximum size in MB for each PDF file
        """
        self.markdown_dir = Path(markdown_dir)
        self.output_dir = Path(output_dir)
        self.size_limit_mb = size_limit_mb
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_markdown_to_pdf(self, markdown_file: Path) -> Tuple[Path, bool]:
        """Convert a single markdown file to PDF using WeasyPrint.
        
        Args:
            markdown_file: Path to the markdown file
            
        Returns:
            Tuple[Path, bool]: (Path to output PDF, Success status)
        """
        pdf_file = self.output_dir / f"{markdown_file.stem}_temp.pdf"
        try:
            # Read and convert markdown to HTML
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            html_content = markdown2.markdown(
                markdown_content,
                extras=['fenced-code-blocks', 'tables', 'header-ids']
            )
            
            # Add basic styling
            css = CSS(string='''
                body { 
                    margin: 2cm;
                    font-family: -apple-system, system-ui, sans-serif;
                    font-size: 12pt;
                    line-height: 1.5;
                }
                pre {
                    background: #f8f8f8;
                    border: 1px solid #ddd;
                    border-radius: 3px;
                    padding: 1em;
                    overflow-x: auto;
                }
                code {
                    font-family: "SF Mono", Consolas, monospace;
                    font-size: 0.9em;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1em 0;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 0.5em;
                }
                th {
                    background: #f5f5f5;
                }
            ''')
            
            # Create PDF with WeasyPrint
            font_config = FontConfiguration()
            html = HTML(string=f"<html><body>{html_content}</body></html>")
            html.write_pdf(
                pdf_file,
                stylesheets=[css],
                font_config=font_config
            )
            return pdf_file, True
            
        except Exception as e:
            print(f"Error converting {markdown_file}: {e}")
            return pdf_file, False
    
    def convert_all_markdown(self) -> List[Path]:
        """Convert all markdown files in the input directory to PDFs.
        
        Returns:
            List[Path]: List of paths to successfully converted PDFs
        """
        markdown_files = sorted(self.markdown_dir.glob('*.md'))
        if not markdown_files:
            print("No markdown files found to convert to PDF")
            return []
        
        temp_pdfs = []
        for md_file in tqdm(markdown_files, desc='Converting markdown files to PDF'):
            pdf_file, success = self.convert_markdown_to_pdf(md_file)
            if success:
                temp_pdfs.append(pdf_file)
        
        return temp_pdfs
    
    def merge_pdfs(self, pdf_files: List[Path], target_chunks: int) -> List[Path]:
        """Merge PDF files while respecting size limits.
        
        Args:
            pdf_files: List of PDF files to merge
            target_chunks: Target number of output files
            
        Returns:
            List[Path]: List of paths to merged PDF files
        """
        if not pdf_files:
            print("No PDFs to merge")
            return []
        
        # Get file sizes and sort by size (largest first)
        pdf_sizes = [(pdf, pdf.stat().st_size / (1024 * 1024)) for pdf in pdf_files]
        pdf_sizes.sort(key=lambda x: x[1], reverse=True)
        
        output_pdfs = []
        current_merger = PdfMerger()
        current_size = 0
        current_index = 1
        
        # Process PDFs, creating new files when size limit is reached
        for pdf_file, size_mb in tqdm(pdf_sizes, desc='Merging PDFs'):
            # If adding this file would exceed size limit, save current file and start new one
            if current_size + size_mb > self.size_limit_mb and current_size > 0:
                output_file = self.output_dir / f'analysis_part_{current_index}.pdf'
                current_merger.write(str(output_file))
                current_merger.close()
                output_pdfs.append(output_file)
                print(f"Created {output_file} ({current_size:.1f}MB)")
                
                # Start new file
                current_merger = PdfMerger()
                current_size = 0
                current_index += 1
            
            # Add current PDF to merger
            current_merger.append(str(pdf_file))
            current_size += size_mb
        
        # Save final file if it has any content
        if current_size > 0:
            output_file = self.output_dir / f'analysis_part_{current_index}.pdf'
            current_merger.write(str(output_file))
            current_merger.close()
            output_pdfs.append(output_file)
            print(f"Created {output_file} ({current_size:.1f}MB)")
        
        return output_pdfs
    
    def cleanup_temp_files(self, temp_files: List[Path]) -> None:
        """Clean up temporary PDF files.
        
        Args:
            temp_files: List of temporary files to remove
        """
        for pdf in temp_files:
            pdf.unlink(missing_ok=True)
    
    def generate_pdfs(self, num_chunks: int) -> List[Path]:
        """Generate PDFs from markdown files.
        
        Args:
            num_chunks: Target number of PDF files to create
            
        Returns:
            List[Path]: List of paths to generated PDF files
        """
        # Convert markdown to PDFs
        temp_pdfs = self.convert_all_markdown()
        if not temp_pdfs:
            return []
        
        # Merge PDFs
        output_pdfs = self.merge_pdfs(temp_pdfs, num_chunks)
        
        # Clean up temporary files
        self.cleanup_temp_files(temp_pdfs)
        
        # Report results
        print(f"\nCreated {len(output_pdfs)} PDF files with size limit of {self.size_limit_mb}MB")
        for pdf in output_pdfs:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            print(f"  {pdf.name}: {size_mb:.1f}MB")
        
        return output_pdfs
