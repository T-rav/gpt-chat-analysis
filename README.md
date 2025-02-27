# GPT Conversation Analysis

A tool for analyzing ChatGPT conversations against the `AI Decision Loop for GenAI`, generating chat analysis reports to explore interaction patterns. This helps you understand AI usage patterns and evaluate the effectiveness of your conversations. Additionally, it still retains the original heatmap capabilities.

---

## AI Decision Loop

🚀 **The 5 Steps (with Adaptive Loops)**
1. **Frame the Decision Context & Guide AI Prompting**
2. **Generate Multi-Perspective AI Outputs & Validate Responses**
3. **Apply Human Judgment & Adjust AI Interaction**
4. **Test for Bias & Feasibility**
5. **Refine, Iterate, and Automate Where Possible**

🔁 **Adaptive Loops:** At any step, return to previous steps to refine prompts, reassess assumptions, or mitigate biases.

---

## Features

### 🧠 AI Intelligence Metrics
- **AI as Critic**: Tracks when AI acts as a critical evaluator
- **Decision Intelligence**: Measures AI-driven decision-making effectiveness
- **Success Rate Analysis**: Evaluates completion rates with AI partnership

### 📊 Conversation Analysis
- Analyzes conversations using OpenAI's GPT-4.
- Evaluates decision-making and interaction patterns.
- Tracks AI's role as a critic and decision intelligence.
- Generates detailed markdown reports.

### 🔥 Heatmap Generation
- Visualizes:
  - Conversation frequency by day.
  - Usage patterns over time.
  - Interaction hotspots.

### 📝 Analysis Output
- Generates markdown reports with:
  - Interaction analysis.
  - Decision-making evaluation.
  - Pattern identification.
  - Recommendations for improvement.

```bash
# Initial analysis to build summaries 
python app.py -o analysis

# Run trend analysis
python app.py --trends analysis
```

### ✅ Format Verification
- Ensures markdown files meet format requirements:
  - Brief Summary
  - Five-Step Decision Loop Analysis
  - Loop Completion Analysis
  - Breakdown & Collaborative Pattern Analysis
  - Recommendations
- Invalid files are removed.

```bash
# Verify files
python app.py --verify-format

# Verify files in a non-default location
python app.py --verify-format -o /custom/dir
```

### 📤 Chat Export
- Export chat histories in:
  - **Text format** (role-separated with timestamps).
  - **JSON format** (full conversation data).
- Output stored in the `exports/` directory.

```bash
# Export chat in text format
python app.py --export-chat <conversation_id> --export-format txt

# Export chat in JSON format
python app.py --export-chat <conversation_id> --export-format json
```

### 📄 PDF Generation
- Converts markdown reports to PDFs.
- Splits large reports into multiple PDFs (default 1MB each). Feed into notebooklm or other chat sessions.

```bash
# Generate PDFs targeting 3 pdfs into a custom location
python app.py --pdf 3 --pdf-dir /custom/dir

# Generate PDFs with a size limit of 4MB into the default location
python app.py --pdf-size-limit 4
```

### ⏳ Incremental Analysis
- Preserves existing work by skipping previously analyzed conversations.
- Processes only new conversations.
- Filters by date with `-d YYYY-MM-DD`.
- Analyze single conversations with `--chat-id` for debugging.

```bash
# Analyze all conversations in a directory
python app.py --trends analysis/
# OR using the original format
python app.py -o analysis

# Analyze a single conversation's markdown file (useful for debugging)
python app.py --trends analysis/ --chat-id <conversation-id>

# OR analyze a single chat session
python app.py -o analysis --chat-id <conversation-id>
```

## Testing

The project uses pytest for testing. Tests are organized into separate files by functionality:
- `test_cli_and_options.py`: Tests for CLI argument parsing
- `test_conversation_analysis.py`: Tests for chat analysis functionality
- `test_conversation_export.py`: Tests for chat export features
- `test_file_validation.py`: Tests for markdown file validation
- `test_pdf_generation.py`: Tests for PDF generation
- `test_trend_analysis.py`: Tests for trend analysis

To run the tests:

```bash
# Run all tests
python -m pytest tests/

# Run tests with output
python -m pytest -v tests/

# Run a specific test file
python -m pytest tests/test_conversation_analysis.py

# Run a specific test
python -m pytest tests/test_conversation_analysis.py::test_analyze_chat
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gpt-chat-analysis.git
   cd gpt-chat-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file:
     ```env
     OPENAI_API_KEY=your_api_key_here
     ```

4. Prepare your data:
   - Place exported ChatGPT conversations (`JSON format`) in the `chats/` directory.

---

## Usage

### 🚀 Basic Commands
```bash
# Run chat analysis
python app.py

# Generate heatmap visualizations
python app.py --heatmap

# Generate trend analysis across all chat analysis documents
python app.py --trends
```

### 📄 PDF Generation
```bash
# Generate PDFs with custom size limit
python app.py --pdf 10 --pdf-size-limit 5
```

### 🔧 Configuration
Customize settings in `configuration.py`:
- `CONVO_FOLDER`: Chat file location.
- `DEFAULT_MODEL`: GPT model (default: `gpt-4o`).
- `MAX_WORKERS`: Number of parallel analysis threads.

Each report includes:
1. **Brief Summary** – Overview of objectives and approach.
2. **Five-Step Decision Loop Analysis** – Step-by-step evaluation.
3. **Collaborative Pattern Analysis** – Observed and novel patterns.
4. **Recommendations** – Actionable improvements.

---

## Contributing

Contributions are welcome! Please submit a Pull Request or open an issue to discuss improvements.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Inspired by [Chip Huyen's AIE Book](https://github.com/chiphuyen/aie-book)
- Uses OpenAI's GPT models for analysis.

