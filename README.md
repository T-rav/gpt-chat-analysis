# GPT Conversation Analysis

A tool for analyzing ChatGPT conversations against my `AI Decision Loop for GenAI` and generating heatmaps to visualize interaction patterns. This tool helps you understand your AI usage patterns and evaluate the effectiveness of your conversations.

## AI Decision Loop
🚀The 5 Steps (Each with Adaptive Loops)
 1. Frame the Decision Context & Guide AI Prompting
 2. Generate Multi-Perspective AI Outputs & Validate Responses
 3. Apply Human Judgment & Adjust AI Interaction
 4. Test for Bias & Feasibility
 5. Refine, Iterate, and Automate Where Possible
 * Adaptive Loops: At any step, return to prior steps as needed to fine-tune prompts, recheck assumptions, or mitigate new biases.

## Features

### Conversation Analysis

![image](https://github.com/user-attachments/assets/ce03be32-5af5-4f62-a4c7-57ccd331d4c9)

- Analyzes chat conversations using OpenAI's GPT-4
- Evaluates decision-making processes and interaction patterns
- Generates detailed markdown reports for each conversation

### Heatmap Generation

![image](https://github.com/user-attachments/assets/000772ed-7754-4ee3-84b7-0a2cd5f3aaa3)

- Creates visual heatmaps showing:
  - Conversation frequency by day
  - Usage patterns over time
  - Interaction hotspots

### Analysis Output
- Markdown reports for each conversation including:
  - Interaction analysis
  - Decision-making evaluation
  - Pattern identification
  - Recommendations for improvement

### PDF Generation
- Automatically converts markdown reports to PDFs
- Manages file size by splitting into multiple PDFs if needed
- Default size limit is 1MB per PDF file
- Will create as many PDFs as needed to include all content
- Warns if more PDFs are created than initially targeted
- Each PDF includes clear file headers showing source markdown files

### Incremental Analysis
The tool supports incremental analysis, preserving existing work:
- Skip already analyzed conversations (based on matching IDs)
- Only process new conversations
- Filter conversations by date using the `-d` option (e.g., `-d 2024-01-01` to analyze only conversations after January 1st, 2024)
- Keep all existing analysis files

Example workflow:
```bash
# Initial analysis
python app.py -o analysis

# Later, add more chats and run again
# (existing files in analysis/ are preserved)
python app.py -o analysis
```

## Requirements

- Python 3.11+
- OpenAI API key
- ChatGPT conversation exports (JSON format)
- Required Python packages (see requirements.txt)

### Exporting ChatGPT Conversations

1. Go to [chat.openai.com](https://chat.openai.com)
2. Click your profile picture (bottom-left) → Settings → Data controls
3. Click 'Export' under 'Export data'
4. Wait for the email and download the .zip file
5. Extract the downloaded zip file
6. Rename the extracted folder to `chats`
7. Move the `chats` folder into your project directory

Example:
```bash
# Assuming you downloaded to ~/Downloads/chatgpt-export.zip
unzip ~/Downloads/chatgpt-export.zip -d ~/Downloads/chatgpt-temp
mv ~/Downloads/chatgpt-temp/conversations chats
mv chats /path/to/your/gpt-heatmap/
```

Note: The export contains all your ChatGPT conversations and related media.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gpt-heatmap.git
   cd gpt-heatmap
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```env
     OPENAI_API_KEY=your_api_key_here
     ```

4. Prepare your data:
   - Create a `chats` directory in the project root
   - Place your ChatGPT conversation JSON files in the `chats` directory

## Usage

### Basic Usage
```bash
# Run chat analysis
python app.py

# Run analysis with custom output directory
python app.py -o custom_output_dir

# Generate heatmap visualizations
python app.py --heatmap
```

### PDF Generation

Generate PDFs from analysis files for easier sharing:

```bash
# Generate PDFs (3 files)
python app.py --pdf 3

# Generate PDFs with custom size limit (10 files, 5MB per file)
python app.py --pdf 10 --pdf-size-limit 5
```

Example output:
```
Generating 10 PDF files from existing markdown...
Converting markdown files to PDF: 100%|████████████| 2285/2285 [03:05<00:00, 12.30it/s]
Merging PDFs: 100%|████████████| 2285/2285 [00:28<00:00, 81.47it/s]

Created 4 PDF files with size limit of 10.0MB
  analysis_part_1.pdf: 13.4MB
  analysis_part_2.pdf: 13.8MB
  analysis_part_3.pdf: 13.9MB
  analysis_part_4.pdf: 6.9MB
```

Options:
- `-o, --output`: Output directory for analysis files (default: 'analysis')
- `-d, --date`: Start date for analysis (format: YYYY-MM-DD). If not provided, analyzes all conversations
- `--pdf N`: Number of PDF files to generate
- `--pdf-size-limit MB`: Maximum size per PDF (default: 1MB)

Requires WeasyPrint. On macOS:
```bash
brew install python-tk cairo pango gdk-pixbuf libffi
```

For other OS, see [WeasyPrint's installation guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation).

### Configuration

Customize analysis settings in `configuration.py`:
- `CONVO_FOLDER`: Chat files location
- `RESEARCH_FOLDER`: Output directory
- `DEFAULT_MODEL`: GPT model (default: 'gpt-4')
- `DEFAULT_TEMPERATURE`: Model temperature (default: 0.62)
- `MAX_WORKERS`: Parallel analysis threads

Each markdown file contains:

1. Brief Summary
   - Overview of objectives and approach

2. Five-Step Decision Loop Analysis
   - Problem Framing & Initial Prompting
   - Response Evaluation & Validation
   - Expertise Application
   - Critical Assessment
   - Process Improvement
   - Overall Assessment

3. Collaborative Pattern Analysis
   - Observed Patterns
   - Novel Patterns

4. Recommendations
   - Improvement suggestions
   - Action steps
   - Strategic adjustments

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by [Chip Huyen's AIE Book](https://github.com/chiphuyen/aie-book)
- Uses OpenAI's GPT models for analysis

## Requirements

- Python 3.6+
- OpenAI API key (for analysis feature)
- Required Python packages (install via pip):
  - matplotlib
  - numpy
  - openai
  - python-dotenv
  - tqdm

## Contributing

Feel free to open issues or submit pull requests with improvements!

