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
- Analyzes chat conversations using OpenAI's GPT-4
- Evaluates decision-making processes and interaction patterns
- Generates detailed markdown reports for each conversation

### Heatmap Generation
- Creates visual heatmaps showing:
  - Conversation frequency by day and hour
  - Usage patterns over time
  - Interaction hotspots

### Analysis Output
- Markdown reports for each conversation including:
  - Interaction analysis
  - Decision-making evaluation
  - Pattern identification
  - Recommendations for improvement

## Requirements

- Python 3.11+
- OpenAI API key
- ChatGPT conversation export (JSON format)
- Required Python packages (see requirements.txt)

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

# Run chat analysis with custom output directory
python app.py -o custom_output_dir

# Generate heatmap visualizations
python app.py --heatmap
```

### Configuration

You can customize the analysis by modifying `config.py`:
- `CONVO_FOLDER`: Location of conversation files
- `RESEARCH_FOLDER`: Default output directory
- `DEFAULT_MODEL`: GPT model to use (default: 'gpt-4')
- `DEFAULT_TEMPERATURE`: Model temperature (default: 0.62)
- `MAX_WORKERS`: Number of parallel analysis threads

## Output Structure

```
output_directory/
├── conversation_id1.md
├── conversation_id2.md
└── ...
```

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

