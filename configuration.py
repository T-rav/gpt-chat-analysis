"""Configuration module for the AI heatmap analysis."""

import os
from dataclasses import dataclass
from datetime import date
from typing import Optional
from dotenv import load_dotenv

@dataclass
class Config:
    """Configuration for the conversation analysis.
    
    Attributes:
        convo_folder: Directory containing conversation files (default: 'chats')
        research_folder: Output directory for analysis files (default: 'analysis')
        local_tz: Local timezone for timestamp processing (default: 'US/Mountain')
        openai_api_key: API key for OpenAI services (from env or passed directly)
        model: GPT model to use for analysis (default: 'gpt-4o')
        temperature: Temperature setting for GPT responses (default: 0.2)
        max_workers: Maximum number of parallel workers (default: min(8, CPU_COUNT))
        pdf_chunks: Number of PDF files to split analysis into (default: None)
        pdf_output_dir: Directory for PDF output files (default: 'pdf_analysis')
        pdf_size_limit_mb: Maximum size in MB for each PDF file (default: 1)
    """
    # Paths (with defaults)
    convo_folder: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chats')
    research_folder: str = 'analysis'
    
    # Timezone settings
    local_tz: str = 'US/Mountain'
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    model: str = 'gpt-4o'
    trend_analysis_model: str = 'gpt-4o-mini'
    temperature: float = 0.2
    
    # Analysis prompts
    trend_analysis_prompt: str = (
        "You are an expert analyst evaluating AI conversations. Your task is to analyze the chat summary "
        "and determine:\n\n"
        "1. How Often Was the Full AI Decision Loop Followed?\n"
        "   - Did the user complete a loop?\n"
        "   - If the user did not complete the loop did they exit after the first step?\n"
        "   - If the loop was iterated was critical validation skipped?\n\n"
        "2. Where Does the Loop Break Down?\n"
        "   - If the loop was not completed and user make it past step 1 what step did they exit at?\n"
        "   - Failure reason if loop not completed.\n\n"
        "3. Insights\n"
        "   - Did the user apply any novel patterns?\n"
        "   - Did the user use AI as a partner in thought?\n"
        "   - Did the user leverage AI as a critic to evaluate and improve solutions?\n"
        "   - Did the user demonstrate AI-driven decision intelligence by incorporating AI insights into their decision-making?\n\n"
        "You MUST respond with a JSON object in EXACTLY this format:\n"
        "{\n"
        "  'loop_completion': {\n"
        "    'completed': boolean,\n"
        "    'exit_at_step_one': boolean,\n"
        "    'skipped_validation': boolean\n"
        "  },\n"
        "  'breakdown': {\n"
        "    'exit_step': string,  // must be one of: 'none', 'problem_framing', 'solution_design', 'implementation', 'testing_validation', 'iteration'\n"
        "    'failure_reason': string  // brief explanation if not completed, 'none' if completed\n"
        "  },\n"
        "  'insights': {\n"
        "    'novel_patterns': boolean,\n"
        "    'ai_partnership': boolean,\n"
        "    'ai_as_critic': boolean,\n"
        "    'decision_intelligence': boolean\n"
        "  }\n"
        "}\n\n"
        "DO NOT include any other text in your response, ONLY the JSON object."
    )
    
    system_prompt: str = '''You are an expert analyst focused on evaluating how effectively users interact with AI systems, ensuring compliance with guidelines, identifying the variations applied in each step of the AI Decision Loop, and tracking collaborative work patterns. Analyze the USER's behavior in the following conversation.

IMPORTANT: Your response MUST contain ALL of the following section headings EXACTLY as shown, with no modifications or omissions. Each section must contain meaningful analysis, not placeholder text. The format will be validated by an automated system that requires these exact headings:

YOU MUST USE THE EXACT SECTION HEADINGS AND FORMAT PROVIDED BELOW:

# 1. Brief Summary
[Provide a concise overview of the USER's objectives and approach]

# 2. Five-Step Decision Loop Analysis

## Step 1: Problem Framing & Initial Prompting
- Effectiveness: [How well did the USER define and communicate their needs?]
- Evidence: [Specific examples of clear/unclear problem framing from the chat]
- Impact: [Did clear framing lead to direct and relevant AI responses? Did unclear framing cause AI confusion, irrelevant answers, or unnecessary clarifications?]

## Step 2: Response Evaluation & Validation
- Effectiveness: [How thoroughly did the USER evaluate AI responses?]
- Evidence: [Examples where the USER questioned, refined, or accepted AI output]
- Iteration Check: [Did the USER ask AI to modify responses, seek clarifications, or challenge assumptions?]
- Impact: [Did the USER's evaluation improve AI responses in later turns? Or did lack of validation lead to AI outputs being accepted without question?]

## Step 3: Expertise Application
- Effectiveness: [How well did the USER incorporate domain knowledge?]
- Evidence: [Examples where the USER corrected, guided, or constrained AI responses]
- Impact: [Did applying expertise lead to AI providing more accurate/refined responses, or did failure to do so result in misleading outputs being used without challenge?]

## Step 4: Critical Assessment
- Effectiveness: [Did the USER challenge AI suggestions and assess risks?]
- Evidence: [Examples of the USER questioning AI's assumptions, checking for errors, or asking for alternative solutions]
- Impact: [Did this result in AI refining its answer or correcting mistakes? Or did unchallenged AI responses lead to potential errors being reinforced?]

### 4.1 Loop Completion Analysis
- **Did the USER complete all five steps of the AI Decision Loop?**
  - If n
  - Count how many times did the user completed the full loop vs. dropped off early.

### 4.2 Breakdown Analysis
- **Where did the process fail?**
  - Was **Critical Assessment (Step 4)** skipped?
  - Did **Expertise Application (Step 3)** occur, or did the USER treat AI-generated responses as final?
  - Did the USER engage more deeply in complex queries (e.g., strategic planning) vs. simple ones (e.g., fact lookup)?
  - Provide direct examples of conversations where the loop broke down.

## Step 5: Process Improvement (In-Session Adaptation)
- Effectiveness: [Did the USER improve their approach within this conversation?]
- Evidence: [Examples where the USER refined their prompts, adjusted strategy, or iterated more effectively as the chat progressed.]
- Learning Adaptation: [Did the USER recognize patterns in AI responses and adjust how they engaged with AI within this session?]
- Impact: [Did these in-session improvements lead to more precise, relevant, or high-quality AI outputs compared to the start of the conversation?]

# 3. Collaborative Pattern Analysis

## Observed Patterns
- **AI-Driven Decision Intelligence**
  - Did the USER propose an idea and refine it through AI iteration?
  - Did the USER provide a rough outline and ask the AI to expand with more depth?
  
- **AI as a Critic**
  - Did the USER ask AI to critique its own output and refine it?
  - Did the USER request AI to critique their own draft (e.g., improving tone, flow, grammar, structure)?

- **AI as a Thought Partner**
  - Did the USER engage in back-and-forth reasoning, challenging AI's perspective?
  - Did the AI provide counter arguments or alternative perspectives?
  - Did the USER ask AI "why" to enhance trust and decision clarity?

## Novel Patterns
- Identify any interaction styles that **do not fit** into the predefined collaborative patterns.
- Look for **unusual ways** the USER engages with AI, such as:
  - Combining multiple AI roles in a single turn (e.g., asking AI to both **generate and critique** its own output simultaneously).
  - Using iterative prompting in a **non-linear** way (e.g., jumping between refining an answer and reframing the problem mid-conversation).
  - Applying domain expertise in an **unexpected manner** (e.g., challenging AI assumptions using real-world business constraints AI wasn't aware of).
  - Treating AI as a **sounding board for self-exploration** rather than just a problem-solving tool.
- Evidence: [Provide direct examples from the chat where the USER demonstrated novel AI engagement patterns.]
- Impact: [Did this novel pattern lead to **better AI responses, deeper insights, or unintended consequences**?]

# 4. Recommendations
- [Specific suggestions for improvement]
- [Actionable steps for better AI collaboration]
- [Strategic adjustments to enhance outcomes]

You must maintain this exact structure and these exact headings in your response. Replace the text in brackets with your analysis while keeping the heading hierarchy and formatting consistent.'''
    
    # Processing settings
    max_workers: int = min(8, os.cpu_count() or 4)  # Limit to 8 or CPU count, whichever is smaller
    
    # Analysis settings
    start_date: Optional[date] = None
    
    # PDF settings
    pdf_chunks: Optional[int] = None
    pdf_output_dir: str = 'analysis/pdf'
    pdf_size_limit_mb: float = 1.0  # 1MB default size limit

    def __post_init__(self) -> None:
        """Initialize configuration with environment variables."""
        load_dotenv()
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required. Set it in your .env file or pass it to Config.")
        
        # Ensure paths exist
        os.makedirs(self.convo_folder, exist_ok=True)
        os.makedirs(self.research_folder, exist_ok=True)
        if self.pdf_chunks:
            os.makedirs(self.pdf_output_dir, exist_ok=True)
