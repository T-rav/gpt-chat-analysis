# GPT Heatmap & Analysis

Generate GitHub-style heatmaps for your ChatGPT conversations and perform in-depth conversation analysis. This tool helps you visualize your ChatGPT usage patterns and understand the quality of your AI interactions through the lens of the 5-part AI Decision Loop framework.

Inspired by [Chip Huyen's AIE Book](https://github.com/chiphuyen/aie-book)

## Features

- **Conversation Heatmaps**: Generate GitHub-style heatmaps showing your ChatGPT usage patterns by year
- **Conversation Analysis**: Analyze your chat conversations using GPT-4 to evaluate how well they follow the 5-part AI Decision Loop framework
- **Markdown Export**: Export your conversations to markdown format for easy reading and sharing

## Setup

1. Clone the repository
2. Place your ChatGPT conversation JSON file in a `chats` directory within the project folder
3. If you want to use the analysis feature, create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the script with different operations:

```bash
# Run both analysis and heatmaps (default)
python ai-heatmap.py

# Generate only the heatmaps
python ai-heatmap.py --operation maps
# or
python ai-heatmap.py -o maps

# Run only the conversation analysis
python ai-heatmap.py --operation analysis
# or
python ai-heatmap.py -o analysis
```

## Output

- **Heatmaps**: Generated as interactive plots showing your ChatGPT usage patterns by year
- **Analysis**: Saved in the `analysis` directory as markdown files
  - Each conversation gets its own analysis file (`analysis_[chat_id].md`)
  - Analysis includes evaluation of:
    - Decision context and prompt quality
    - Multi-perspective generation and validation
    - Human judgment integration
    - Bias and feasibility testing
    - Iteration and automation opportunities

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

