# GPT Heatmap & Conversation Analysis

An advanced tool for analyzing ChatGPT conversations using the 5-step AI Decision Loop framework and collaborative work patterns. This tool helps you understand and improve your AI interactions by evaluating both the decision-making process and the effectiveness of human-AI collaboration.

## Features

### Conversation Analysis
- **5-Step AI Decision Loop Evaluation**:
  1. Frame the Decision Context & Guide AI Prompting
  2. Generate Multi-Perspective AI Outputs & Validate
  3. Apply Human Judgment & Adjust AI Interaction
  4. Test for Bias & Feasibility
  5. Refine, Iterate, and Automate

### Collaborative Pattern Analysis
- **Known Patterns**:
  1. Iterative Refinement (Human ↔ AI refinement cycles)
  2. Review and Adjustment (Quality checks and critiques)
  3. Reasoning and Challenge (Thought process exploration)
- **Pattern Discovery**: Identifies and documents new effective collaboration patterns

### Analysis Output
- Detailed markdown reports for each conversation including:
  - Summary of the interaction
  - Evaluation of decision loop steps
  - Collaborative pattern analysis
  - Specific examples and recommendations
  - Novel pattern discoveries

## Requirements

- Python 3.8+
- OpenAI API key
- ChatGPT conversation export (JSON format)

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
# Run analysis with default settings
python ai-heatmap.py

# Specify custom output directory
python ai-heatmap.py -o custom_output_dir
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
1. Conversation Summary
2. Decision Loop Analysis
3. Collaboration Pattern Analysis
4. Improvement Recommendations
5. Novel Pattern Discoveries

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

