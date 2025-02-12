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
    
    def __init__(self, markdown_dir: str, output_dir: str, size_limit_mb: float = 1.0):
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
            # Read and preprocess markdown content
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
                
            # Clean up markdown content
            markdown_content = markdown_content.strip()  # Remove leading/trailing whitespace
            markdown_content = markdown_content.replace('%', '')  # Remove stray % characters
            markdown_content = '\n'.join(line.rstrip() for line in markdown_content.splitlines())  # Clean line endings
            html_content = markdown2.markdown(
                markdown_content,
                extras=[
                    'fenced-code-blocks',
                    'tables',
                    'header-ids',
                    'break-on-newline',
                    'cuddled-lists',
                    'markdown-in-html',
                    'smarty-pants',
                    'target-blank-links',
                    'html-classes',  # Allow HTML class attributes
                    'code-friendly'  # Better code block handling
                ]
            )
            
            # Add basic styling with improved markdown formatting
            css = CSS(string='''
                body { 
                    margin: 2cm;
                    font-family: -apple-system, system-ui, sans-serif;
                    font-size: 12pt;
                    line-height: 1.6;
                    color: #333;
                    max-width: 50em;
                    margin: 2cm auto;
                }
                h1, h2, h3, h4, h5, h6 {
                    margin-top: 1.5em;
                    margin-bottom: 0.5em;
                    color: #222;
                    font-weight: 600;
                    line-height: 1.3;
                }
                h1 { font-size: 2em; border-bottom: 2px solid #eee; padding-bottom: 0.3em; }
                h2 { font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }
                h3 { font-size: 1.3em; }
                p { margin: 1em 0; }
                ul, ol { 
                    margin: 1em 0;
                    padding-left: 2em;
                }
                li { margin: 0.5em 0; }
                li > p { margin: 0.3em 0; }
                pre {
                    background: #f8f8f8;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 1em;
                    margin: 1em 0;
                    overflow-x: auto;
                    font-size: 0.9em;
                    line-height: 1.5;
                }
                code {
                    font-family: monospace;
                    font-size: 8pt;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 0.3em 0;
                    font-size: 8pt;
                }
                th, td {
                    padding: 3px;
                    text-align: left;
                }
                blockquote {
                    margin: 0.5em 0;
                    padding-left: 1em;
                    color: #666;
                    border-left: 2px solid #ddd;
                }
                hr {
                    border: none;
                    border-top: 1px solid #eee;
                    margin: 1em 0;
                }
            ''')
            
            # Create optimized PDF with compression
            font_config = FontConfiguration()
            try:
                html = HTML(string=f"<html><body>{html_content}</body></html>")
                # Create minimal PDF with maximum compression
                doc = html.render(
                    stylesheets=[css],
                    font_config=font_config,
                    optimize_size=('fonts', 'images'),
                    presentational_hints=True,  # Use HTML size hints
                )
                
                doc.write_pdf(
                    target=pdf_file,
                    zoom=0.9,  # Slightly reduce size
                    optimize_images=True,
                    jpeg_quality=70,  # Reduce image quality
                    compress=True,  # Enable compression
                    attachments=[]  # No attachments needed
                )
                
                # Explicitly release resources
                del doc
                del html
                del html_content  # Release the large string
                del font_config
                import gc
                gc.collect()
            except Exception as e:
                raise Exception(f"WeasyPrint error: {e}")
            return pdf_file, True
            
        except Exception as e:
            print(f"Error converting {markdown_file}: {e}")
            return pdf_file, False
    
    def merge_markdown_files(self, markdown_files: List[Path], target_chunks: int) -> List[Path]:
        """Merge markdown files into chunks before converting to PDF.
        
        Args:
            markdown_files: List of markdown files to merge
            target_chunks: Target number of output files initially
            
        Returns:
            List[Path]: List of paths to merged markdown files
        """
        if not markdown_files:
            return []
            
        merged_files = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        # PDF files with styling can be much larger than markdown, use very conservative factor
        # Calculate markdown size limit accounting for PDF overhead
        pdf_overhead_factor = 1.3   # PDFs are ~30% larger than markdown
        md_size_limit = (self.size_limit_mb / pdf_overhead_factor) * 1024 * 1024
        
        with tqdm(total=len(markdown_files), desc='Merging markdown files') as pbar:
            for md_file in markdown_files:
                try:
                    file_size = md_file.stat().st_size
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            # If this file would exceed size limit, save current chunk and start new one
                            if current_size + file_size > md_size_limit and current_chunk:
                                merged_file = self.output_dir / f"merged_part_{chunk_index}.md"
                                with open(merged_file, 'w', encoding='utf-8') as f:
                                    f.write('\n\n---\n\n'.join(current_chunk))
                                merged_files.append(merged_file)
                                current_chunk = []
                                current_size = 0
                                chunk_index += 1
                            
                            # Add file to current chunk
                            separator = "<div style='font-size: 8pt; color: #666;'>======================</div>"
                            header = (
                                f"\n\n{separator}\n"  # Top separator
                                f"<div style='font-size: 8pt; color: #666;'>File: {md_file.name}</div>\n"  # Filename in smaller font
                                f"{separator}\n\n"  # Bottom separator
                            )
                            current_chunk.append(header + content)
                            current_size += file_size
                except Exception as e:
                    print(f"\nError reading {md_file}: {e}")
                pbar.update(1)
            
            # Save final chunk if it has content
            if current_chunk:
                merged_file = self.output_dir / f"merged_part_{chunk_index}.md"
                with open(merged_file, 'w', encoding='utf-8') as f:
                    f.write('\n\n---\n\n'.join(current_chunk))
                merged_files.append(merged_file)
        
        if len(merged_files) > target_chunks:
            print(f"\nNote: Created {len(merged_files)} files to stay under size limit of {self.size_limit_mb}MB per file (original target was {target_chunks} files)")
        
        return merged_files
        
        return merged_files
    
    def convert_all_markdown(self) -> List[Path]:
        """Convert all markdown files in the input directory to PDFs.
        
        Returns:
            List[Path]: List of paths to successfully converted PDFs
        """
        # Get all markdown files and filter out any that don't look like analysis files
        markdown_files = [f for f in sorted(self.markdown_dir.glob('*.md')) 
                         if f.stem and len(f.stem) > 8 and '-' in f.stem]  # Analysis files have UUIDs
        if not markdown_files:
            print("No markdown files found to convert to PDF")
            return []
        
        # First merge markdown files into chunks
        merged_files = self.merge_markdown_files(markdown_files, target_chunks=10)
        if not merged_files:
            print("Failed to merge any markdown files")
            return []
            
        # Convert merged files to PDFs
        temp_pdfs = []
        with tqdm(total=len(merged_files), desc='Converting merged files to PDF') as pbar:
            for md_file in merged_files:
                pdf_file, success = self.convert_markdown_to_pdf(md_file)
                if success:
                    temp_pdfs.append(pdf_file)
                pbar.update(1)
                
                # Clean up merged markdown file
                try:
                    md_file.unlink()
                except Exception:
                    pass
                
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
        # Convert markdown directly to final PDFs
        output_pdfs = self.convert_all_markdown()
        
        # Report results
        if output_pdfs:
            print(f"\nCreated {len(output_pdfs)} PDF files with size limit of {self.size_limit_mb}MB")
            for pdf in output_pdfs:
                size_mb = pdf.stat().st_size / (1024 * 1024)
                print(f"  {pdf.name}: {size_mb:.1f}MB")
            
        return output_pdfs
