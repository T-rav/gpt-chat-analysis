# GPT Conversation Analysis

A research toolbox for analyzing your ChatGPT conversations using the AI Decision Loop framework. Generate quick reports and insights to study and optimize your AI interactions.

---

## AI Decision Loop

🚀 **The 5 Steps**
1. **Frame the Decision Context** – Define constraints, assumptions, and the problem.
2. **AI Output Generation & Thought Partnership** – Treat AI as a collaborator, not a magic box.
3. **Apply Human Judgment** – Challenge AI's responses, ask for justification.
4. **Verify & Validate** – Fact-check for reliability, especially in high-stakes tasks.
5. **Refine & Iterate** – Learn from interactions, improve prompts, and automate selectively.

---

## Installation

**Requirements:**
- Python 3.8 or higher
- pip (Python package installer)

1. Clone the repository:
   ```bash
   git clone https://github.com/T-rav/gpt-chat-analysis.git
   cd gpt-chat-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file:
     ```env
     OPENAI_API_KEY=your_api_key_here
     ```

4. Prepare your data:
   - Place exported ChatGPT conversations in the  root of the project and rename it to `chats`.

---

## Usage

### 🔧 Configuration
Customize settings in `configuration.py`:
- `CONVO_FOLDER`: Chat file location.
- `DEFAULT_MODEL`: GPT model (default: `gpt-4o`).
- `MAX_WORKERS`: Number of parallel analysis threads.

### 🚀 Running the Analysis Pipeline

Follow these steps in order:
```bash
# Run chat analysis saving to the analysis/ folder
python app.py -o analysis

# Verify files after doing analysis summaries
# Note: Removes invalid ones - re-run analysis and it will just reprocess the bad ones
python app.py --verify-format analysis/

# Generate trend analysis across all chat analysis documents
# This will make a bunch of json files
python app.py --trends analysis/

# Export a chat for debugging - supports both txt and json
# Only needed if checking results / debugging issues
python app.py --export-chat <conversation_id> --export-format txt

# Generate PDFs with custom size limit
# Pdfs are only needed if doing `Vibe Analysis` before running trends analysis
python app.py --pdf 10 --pdf-size-limit 5
```
---

## Features

### 🧠 AI Intelligence Metrics
- **Success Rate Analysis**: Evaluates completion rates with AI Decision Loop.
- **AI as a Partner**: Tracks when AI acts as a thought partner.
- **Decision Intelligence**: Measures AI-driven decision-making effectiveness

### 📊 Conversation Analysis
- Analyzes conversations using OpenAI's GPT-4.
- Evaluates decision-making and interaction patterns.
- Tracks AI's role as a collaborative partner.
- Generates detailed markdown reports.

### ✅ Format Verification
- Ensures markdown files meet format requirements:
  - Brief Summary
  - Five-Step Decision Loop Analysis
  - Loop Completion Analysis
  - Breakdown & Collaborative Pattern Analysis
  - Recommendations
- Invalid files are removed.

### 📤 Chat Export
- Export chat histories in:
  - **Text format** (role-separated with timestamps).
  - **JSON format** (full conversation data).
- Output stored in the `exports/` directory.

### 📄 PDF Generation
- Converts markdown reports to PDFs.
- Splits large reports into multiple PDFs (default 1MB each). Feed into notebooklm or other chat sessions.


### ⏳ Incremental Analysis
- Preserves existing work by skipping previously analyzed conversations.
- Processes only new conversations.
- Filters by date with `-d YYYY-MM-DD`.
- Analyze single conversations with `--chat-id` for debugging.

```bash
# Analyze a single conversation's markdown file (useful for debugging)
python app.py --trends analysis/ --chat-id <conversation-id>

# OR analyze a single chat session
python app.py -o analysis --chat-id <conversation-id>
```

## Testing

The project uses pytest for testing. Tests are organized into separate files by functionality.

To run the tests:

```bash
# Run all tests
python -m pytest tests/
```

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

