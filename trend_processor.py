import os
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from typing import Dict, Any, List
from configuration import Config
class TrendProcessor:
    """Handles analysis of markdown files for chat completion statistics.
    
    This class is responsible for:
    1. Reading markdown files from a specified directory
    2. Analyzing loop completion patterns
    3. Generating statistical summaries
    """
    
    def __init__(self, output_dir: str = 'analysis'):
        """Initialize the analysis processor with OpenAI client.
        
        Args:
            output_dir (str): Directory to save analysis JSON files (default: 'analysis')
        """
        config = Config()
        if not config.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.model
        self.temperature = config.temperature
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def _should_process_file(self, md_path: str) -> bool:
        """Check if a markdown file needs to be processed.
        
        Args:
            md_path (str): Path to the markdown file
            
        Returns:
            bool: True if file should be processed, False if it can be skipped
        """
        json_path = os.path.join(
            self.output_dir,
            os.path.splitext(os.path.basename(md_path))[0] + '.json'
        )
        
        if os.path.exists(json_path):
            md_mtime = os.path.getmtime(md_path)
            json_mtime = os.path.getmtime(json_path)
            if json_mtime >= md_mtime:
                return False
        return True

    def _process_file_with_cache(self, filepath: str) -> Dict:
        """Process a file, using cached results if available.
        
        Args:
            filepath (str): Path to the markdown file
            
        Returns:
            dict: Analysis results, either from cache or newly processed
        """
        filename = os.path.basename(filepath)
        
        # Check for cached results
        if not self._should_process_file(filepath):
            print(f"Using cached analysis for {filename}")
            json_path = os.path.join(
                self.output_dir,
                os.path.splitext(filename)[0] + '.json'
            )
            with open(json_path, 'r') as f:
                return json.load(f)
        
        # Process file if no cache available
        stats = self._process_file(filepath)
        
        # Print result
        status = "✓" if stats['completed'] == 1 else "✗"
        exit_info = f" (Exit: {stats['exit_step']})" if not stats['completed'] else ""
        print(f"{filename}: {status}{exit_info}")
        
        return stats

    def analyze_directory(self, directory: str) -> Dict:
        """Analyze markdown files in the specified directory for loop completion.
        
        Args:
            directory (str): Path to directory containing markdown files
            
        Returns:
            dict: Analysis summary with completion statistics
            
        Raises:
            FileNotFoundError: If the specified directory doesn't exist
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        
        # Get list of markdown files
        md_files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith('.md')
        ]
        
        if not md_files:
            print("No markdown files found to analyze")
            return {}
        
        print(f"\nAnalyzing {len(md_files)} files in parallel:")
        
        # Process files in parallel
        stats_list = []
        max_workers = min(mp.cpu_count(), len(md_files))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_file_with_cache, f): f 
                for f in md_files
            }
            
            for future in as_completed(future_to_file):
                try:
                    stats = future.result()
                    stats_list.append(stats)
                except Exception as e:
                    file = future_to_file[future]
                    print(f"Error processing {os.path.basename(file)}: {str(e)}")
        
        return self._generate_summary(stats_list)
    
    def _analyze_with_openai(self, text: str, filename: str) -> Dict[str, Any]:
        """Analyze text using OpenAI to determine loop completion and patterns.
        
        Args:
            text (str): The conversation text to analyze
            filename (str): Name of file being analyzed, for logging
            
        Returns:
            dict: Detailed analysis of the AI Decision Loop execution
        """
        system_prompt = (
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
            "   - Did the user use AI as a partner in thought?\n\n"
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
            "    'ai_partnership': boolean\n"
            "  }\n"
            "}\n\n"
            "DO NOT include any other text in your response, ONLY the JSON object."
        )
        
        user_prompt = f"Analyze this conversation and return ONLY a JSON object according to the specified format:\n\n{text}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean the response - remove markdown code blocks if present
            if result.startswith('```'):
                # Find the first and last ``` and extract content between them
                start = result.find('\n', result.find('```')) + 1
                end = result.rfind('```')
                result = result[start:end].strip()
                
                # If it had a json tag at the start, remove that too
                if result.startswith('json\n'):
                    result = result[5:].strip()
            
            # Handle simple yes/no responses
            if result.lower() in ['yes', 'no']:
                is_completed = result.lower() == 'yes'
                analysis = {
                    'loop_completion': {
                        'completed': is_completed,
                        'exit_at_step_one': not is_completed,
                        'skipped_validation': False
                    },
                    'breakdown': {
                        'exit_step': 'none' if is_completed else 'step_1',
                        'failure_reason': 'none' if is_completed else 'Insufficient information for detailed analysis'
                    },
                    'insights': {
                        'novel_patterns': False,
                        'ai_partnership': False
                    }
                }
            else:
                # Try to parse as JSON
                analysis = json.loads(result)
            
            # Save the analysis
            json_filename = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}.json")
            with open(json_filename, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            return analysis
        except Exception as e:
            default_analysis = {
                'loop_completion': {'completed': False, 'exit_at_step_one': True, 'skipped_validation': False},
                'breakdown': {'exit_step': 'unknown', 'failure_reason': 'Failed to parse analysis'},
                'insights': {'novel_patterns': False, 'ai_partnership': False}
            }
            
            # Save the default analysis for failed cases
            json_filename = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}.json")
            with open(json_filename, 'w') as f:
                json.dump(default_analysis, f, indent=2)
            
            return default_analysis

    def _process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single markdown file and extract completion statistics.
        
        Args:
            file_path (str): Path to the markdown file
            
        Returns:
            dict: Detailed statistics about the conversation
        """
        filename = os.path.basename(file_path)
        print(f"\nProcessing {filename}:")
        
        with open(file_path, "r", encoding="utf-8") as file:
            md_text = file.read()
        
        analysis = self._analyze_with_openai(md_text, filename)
        
        # Extract the summary section if it exists
        import re
        summary_match = re.search(r'# 1\. Brief Summary\n(.+?)\n#', md_text, re.DOTALL)
        summary_text = summary_match.group(1).strip() if summary_match else md_text
        
        # Get the analysis results
        completion = analysis['loop_completion']
        breakdown = analysis['breakdown']
        insights = analysis['insights']
        
        return {
            'total': 1,
            'completed': 1 if completion['completed'] else 0,
            'exit_at_step_one': 1 if completion['exit_at_step_one'] else 0,
            'skipped_validation': 1 if completion['skipped_validation'] else 0,
            'exit_step': breakdown['exit_step'],
            'failure_reason': breakdown['failure_reason'],
            'novel_patterns': 1 if insights['novel_patterns'] else 0,
            'ai_partnership': 1 if insights['ai_partnership'] else 0
        }
    
    def _generate_summary(self, stats_list):
        """Generate a summary of the analysis results.
        
        Args:
            stats_list (list): List of statistics dictionaries from processed files
            
        Returns:
            dict: Aggregated statistics and insights
        """
        total_chats = len(stats_list)
        if total_chats == 0:
            return {"Total Chats Analyzed": 0}
            
        # Initialize counters
        completed = sum(s['completed'] for s in stats_list)
        exit_at_step_one = sum(s['exit_at_step_one'] for s in stats_list)
        skipped_validation = sum(s['skipped_validation'] for s in stats_list)
        novel_patterns = sum(s['novel_patterns'] for s in stats_list)
        ai_partnership = sum(s['ai_partnership'] for s in stats_list)
        
        # Count exit steps
        exit_steps = {}
        failure_reasons = {}
        for s in stats_list:
            if not s['completed']:
                exit_steps[s['exit_step']] = exit_steps.get(s['exit_step'], 0) + 1
                failure_reasons[s['failure_reason']] = failure_reasons.get(s['failure_reason'], 0) + 1
        
        return {
            "Total Chats Analyzed": total_chats,
            "Loop Completion": {
                "Completed (%)": (completed / total_chats) * 100,
                "Exit at Step One (%)": (exit_at_step_one / total_chats) * 100,
                "Skipped Validation (%)": (skipped_validation / total_chats) * 100
            },
            "Breakdown": {
                "Exit Steps": exit_steps,
                "Failure Reasons": failure_reasons
            },
            "Insights": {
                "Novel Patterns (%)": (novel_patterns / total_chats) * 100,
                "AI Partnership (%)": (ai_partnership / total_chats) * 100
            }
        }
