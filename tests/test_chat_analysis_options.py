"""Tests for ChatAnalysisOptions class functionality."""

import os
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from chat_analysis_options import ChatAnalysisOptions
from configuration import Config

@pytest.fixture
def mock_args():
    """Create mock command line arguments."""
    args = MagicMock()
    args.output = "test_output"
    args.pdf = None
    args.pdf_dir = "test_pdfs"
    args.pdf_size_limit = 1.0
    args.date = None
    args.export_chat = None
    args.export_format = "txt"
    args.trends = None
    args.verify_format = False
    args.chat_id = None
    return args

@pytest.fixture
def chat_analysis(mock_args):
    """Create a ChatAnalysisOptions instance with mocked arguments."""
    with patch('chat_analysis_options.CLIParser') as mock_cli:
        mock_cli.parse_args.return_value = mock_args
        return ChatAnalysisOptions()

def test_initialize_config(chat_analysis, mock_args):
    """Test configuration initialization."""
    config = chat_analysis.config
    assert isinstance(config, Config)
    assert config.research_folder == mock_args.output
    assert config.pdf_chunks == mock_args.pdf
    assert config.pdf_output_dir == mock_args.pdf_dir
    assert config.pdf_size_limit_mb == mock_args.pdf_size_limit
    assert config.start_date == mock_args.date

def test_export_chat(chat_analysis, mock_args):
    """Test chat export functionality."""
    mock_args.export_chat = "test_chat"
    mock_args.export_format = "json"
    
    with patch('chat_analysis_options.ConversationData') as mock_conv:
        mock_instance = mock_conv.return_value
        mock_instance.export_chat_history.return_value = "output.json"
        
        chat_analysis.export_chat()
        
        mock_conv.assert_called_once_with(chat_analysis.config)
        mock_instance.export_chat_history.assert_called_once_with("test_chat", "json")

def test_generate_pdfs(chat_analysis, mock_args):
    """Test PDF generation functionality."""
    mock_args.pdf = 5
    mock_args.pdf_dir = "test_pdf_dir"
    
    with patch('chat_analysis_options.PDFGenerator') as mock_pdf:
        chat_analysis.generate_pdfs()
        
        mock_pdf.assert_called_once_with(
            markdown_dir=mock_args.output,
            output_dir=mock_args.pdf_dir,
            size_limit_mb=mock_args.pdf_size_limit
        )
        mock_pdf.return_value.generate_pdfs.assert_called_once_with(5)

def test_verify_markdown_format_valid(chat_analysis):
    """Test markdown verification with valid files."""
    with patch('file_validator.FileValidator') as mock_validator:
        mock_validator.verify_and_clean_md_files.return_value = ([], 0)
        
        chat_analysis.verify_markdown_format()
        
        mock_validator.verify_and_clean_md_files.assert_called_once_with(chat_analysis.args.output)

def test_verify_markdown_format_invalid(chat_analysis):
    """Test markdown verification with invalid files."""
    invalid_files = ["bad1.md", "bad2.md"]
    with patch('file_validator.FileValidator') as mock_validator:
        mock_validator.verify_and_clean_md_files.return_value = (invalid_files, len(invalid_files))
        
        chat_analysis.verify_markdown_format()
        
        mock_validator.verify_and_clean_md_files.assert_called_once_with(chat_analysis.args.output)

def test_analyze_trends_single_chat(chat_analysis, mock_args):
    """Test trend analysis for a single chat."""
    mock_args.chat_id = "test_chat"
    mock_args.trends = "test_trends"
    
    with patch('chat_analysis_options.TrendProcessor') as mock_trend, \
         patch('os.path.isfile', return_value=True):
        mock_instance = mock_trend.return_value
        mock_instance._process_file.return_value = {"completed": 1, "total": 1}
        
        chat_analysis.analyze_trends()
        
        mock_trend.assert_called_once_with(
            output_dir=chat_analysis.config.research_folder,
            force_reprocess=mock_args.force_reprocess
        )
        mock_instance._process_file.assert_called_once_with(os.path.join("test_trends", "test_chat.md"))

def test_analyze_trends_directory(chat_analysis, mock_args):
    """Test trend analysis for a directory."""
    mock_args.trends = "test_trends"
    
    with patch('chat_analysis_options.TrendProcessor') as mock_trend:
        mock_instance = mock_trend.return_value
        mock_instance.analyze_directory.return_value = {
            "Total Chats Analyzed": 5,
            "Loop Completion": {"Completed (%)": 80.0}
        }
        
        chat_analysis.analyze_trends()
        
        mock_trend.assert_called_once_with(
            output_dir=chat_analysis.config.research_folder,
            force_reprocess=mock_args.force_reprocess
        )
        mock_instance.analyze_directory.assert_called_once_with("test_trends")

def test_analyze_chats_single(chat_analysis, mock_args):
    """Test analysis of a single chat."""
    mock_args.chat_id = "test_chat"
    
    with patch('chat_analysis_options.ConversationData') as mock_conv:
        mock_instance = mock_conv.return_value
        
        chat_analysis.analyze_chats()
        
        mock_conv.assert_called_once_with(chat_analysis.config)
        mock_instance.analyze_single_chat.assert_called_once_with("test_chat")

def test_analyze_chats_all(chat_analysis):
    """Test analysis of all chats."""
    with patch('chat_analysis_options.ConversationData') as mock_conv:
        mock_instance = mock_conv.return_value
        
        chat_analysis.analyze_chats()
        
        mock_conv.assert_called_once_with(chat_analysis.config)
        mock_instance.analyze_all_chats_parallel.assert_called_once()

def test_run_verify_format(chat_analysis, mock_args):
    """Test running verify format operation."""
    mock_args.verify_format = True
    
    with patch.object(chat_analysis, 'verify_markdown_format') as mock_verify:
        chat_analysis.run()
        mock_verify.assert_called_once()

def test_run_export_chat(chat_analysis, mock_args):
    """Test running chat export operation."""
    mock_args.export_chat = "test_chat"
    
    with patch.object(chat_analysis, 'export_chat') as mock_export:
        chat_analysis.run()
        mock_export.assert_called_once()

def test_run_generate_pdfs(chat_analysis, mock_args):
    """Test running PDF generation operation."""
    mock_args.pdf = 5
    
    with patch.object(chat_analysis, 'generate_pdfs') as mock_generate:
        chat_analysis.run()
        mock_generate.assert_called_once()

def test_run_analyze_trends(chat_analysis, mock_args):
    """Test running trend analysis operation."""
    mock_args.trends = "test_trends"
    
    with patch.object(chat_analysis, 'analyze_trends') as mock_analyze:
        chat_analysis.run()
        mock_analyze.assert_called_once()

def test_run_analyze_chats(chat_analysis, mock_args):
    """Test running chat analysis operation."""
    with patch.object(chat_analysis, 'analyze_chats') as mock_analyze:
        chat_analysis.run()
        mock_analyze.assert_called_once()

def test_run_error_handling(chat_analysis):
    """Test error handling in run method."""
    with patch.object(chat_analysis, 'analyze_chats', side_effect=ValueError("Test error")):
        with pytest.raises(SystemExit) as exc_info:
            chat_analysis.run()
        assert exc_info.value.code == 1

    with patch.object(chat_analysis, 'analyze_chats', side_effect=FileNotFoundError("Test error")):
        with pytest.raises(SystemExit) as exc_info:
            chat_analysis.run()
        assert exc_info.value.code == 1

    with patch.object(chat_analysis, 'analyze_chats', side_effect=Exception("Test error")):
        with pytest.raises(SystemExit) as exc_info:
            chat_analysis.run()
        assert exc_info.value.code == 1
