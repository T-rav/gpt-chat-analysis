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
            json_path = os.path.join(
                self.output_dir,
                os.path.splitext(filename)[0] + '.json'
            )
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Map old format to new format
                stats = {
                    'completed': 1 if data.get('loop_completion', {}).get('completed', False) else 0,
                    'exit_at_step_one': data.get('loop_completion', {}).get('exit_at_step_one', False),
                    'skipped_validation': data.get('loop_completion', {}).get('skipped_validation', False),
                    'exit_step': data.get('breakdown', {}).get('exit_step', 'unknown'),
                    'failure_reason': data.get('breakdown', {}).get('failure_reason', 'unknown'),
                    'novel_patterns': data.get('insights', {}).get('novel_patterns', False),
                    'ai_partnership': data.get('insights', {}).get('ai_partnership', False),
                    'cached': True
                }
                return stats
        
        # Process file if no cache available
        stats = self._process_file(filepath)
        stats['cached'] = False
        
        # Print result if not cached
        status = "✓" if stats['completed'] == 1 else "✗"
        exit_info = f" (Exit: {stats['exit_step']})" if not stats['completed'] else ""
        print(f"\n{filename}: {status}{exit_info}")
        
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
        
        total_files = len(md_files)
        print(f"\nFound {total_files} files to process")
        
        # Process files in parallel
        stats_list = []
        processed = 0
        cached = 0
        errors = 0
        
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
                    processed += 1
                    if 'cached' in stats and stats['cached']:
                        cached += 1
                    
                    # Print progress
                    print(f"Progress: {processed}/{total_files} files ({cached} cached)", end='\r')
                except Exception as e:
                    file = future_to_file[future]
                    print(f"\nError processing {os.path.basename(file)}: {str(e)}")
                    errors += 1
        
        # Print final stats
        print(f"\nCompleted: {processed} files processed ({cached} from cache, {errors} errors)")
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
            
        # Filter out step one exits first
        engaged_chats = [s for s in stats_list if not s.get('exit_at_step_one', False)]
        total_engaged = len(engaged_chats)
        
        if total_engaged == 0:
            return {
                "Total Chats Analyzed": total_chats,
                "Step One Exits": total_chats,
                "Engaged Conversations": 0
            }
        
        # Initialize counters for engaged chats only
        completed = sum(1 for s in engaged_chats if s.get('completed', 0) == 1)
        skipped_validation = sum(1 for s in engaged_chats if s.get('skipped_validation', False))
        novel_patterns = sum(1 for s in engaged_chats if s.get('novel_patterns', False))
        ai_partnership = sum(1 for s in engaged_chats if s.get('ai_partnership', False))
        
        def normalize_reason(reason):
            """Normalize failure reasons to avoid duplicates with slightly different wording."""
            reason = reason.lower().strip()
            
            # Common variations of the same reason
            if 'insufficient' in reason or 'not enough' in reason:
                return 'insufficient_information'
            if 'unclear' in reason or 'ambiguous' in reason:
                return 'unclear_requirements'
            if 'invalid' in reason or 'malformed' in reason:
                return 'invalid_format'
            if 'timeout' in reason or 'no response' in reason:
                return 'timeout'
            if reason == 'none':
                return 'none'
            if reason == 'unknown':
                return 'unknown'
            
            # Default to the original reason if no match
            return reason.replace(' ', '_')
        
        # Count exit steps for engaged chats only
        exit_steps = {}
        failure_reasons = {}
        for s in engaged_chats:
            if s.get('completed', 0) != 1:
                exit_step = s.get('exit_step', 'unknown')
                failure_reason = normalize_reason(s.get('failure_reason', 'unknown'))
                exit_steps[exit_step] = exit_steps.get(exit_step, 0) + 1
                failure_reasons[failure_reason] = failure_reasons.get(failure_reason, 0) + 1
        
        return {
            "Total Chats": {
                "Total Analyzed": total_chats,
                "Step One Exits": total_chats - total_engaged,
                "Step One Exit Rate (%)": ((total_chats - total_engaged) / total_chats) * 100,
                "Engaged Conversations": total_engaged
            },
            "Loop Completion (of engaged)": {
                "Completed (%)": (completed / total_engaged) * 100,
                "Skipped Validation (%)": (skipped_validation / total_engaged) * 100
            },
            "Breakdown (of engaged)": {
                "Exit Steps": exit_steps,
                "Failure Reasons": failure_reasons
            },
            "Insights (of engaged)": {
                "Novel Patterns (%)": (novel_patterns / total_engaged) * 100,
                "AI Partnership (%)": (ai_partnership / total_engaged) * 100
            }
        }
