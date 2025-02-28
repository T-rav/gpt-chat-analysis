# GPT Conversation Analysis

A research toolbox for analyzing your ChatGPT conversations using the AI Decision Loop framework. Generate quick reports, heatmaps, and insights to study and optimize your AI interactions.

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

## Installation

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
   - Place exported ChatGPT conversations in the `chats/` directory under the git clone directory.

---

## Usage

### 🔧 Configuration
Customize settings in `configuration.py`:
- `CONVO_FOLDER`: Chat file location.
- `DEFAULT_MODEL`: GPT model (default: `gpt-4o`).
- `MAX_WORKERS`: Number of parallel analysis threads.

### 🚀 Basic Commands
```bash
# Run chat analysis
python app.py -o analysis

# Verify files after doing analysis summaries
python app.py --verify-format analysis/

# Generate PDFs with custom size limit
python app.py --pdf 10 --pdf-size-limit 5

# Generate trend analysis across all chat analysis documents
python app.py --trends analysis/

# Export a chat for debugging - supports both txt and json
python app.py --export-chat <conversation_id> --export-format txt

# Generate heatmap visualizations (The OG functionality when cloned)
python app.py --heatmap
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

