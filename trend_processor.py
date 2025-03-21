import os
import json

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
    
    def __init__(self, output_dir: str = 'analysis', force_reprocess: bool = False):
        """Initialize the analysis processor with OpenAI client.
        
        Args:
            output_dir (str): Directory to save analysis JSON files (default: 'analysis')
            force_reprocess (bool): If True, reprocess all files even if cached results exist
        """
        self.config = Config()
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        self.client = OpenAI(api_key=self.config.openai_api_key)
        self.model = self.config.model
        self.temperature = self.config.temperature
        self.output_dir = output_dir
        self.force_reprocess = force_reprocess
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def _should_process_file(self, md_path: str) -> bool:
        """Check if a markdown file needs to be processed.
        
        Args:
            md_path (str): Path to the markdown file
            
        Returns:
            bool: True if file should be processed, False if it can be skipped
        """
        if self.force_reprocess:
            return True
            
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
                    'ai_as_critic': data.get('insights', {}).get('ai_as_critic', False),
                    'decision_intelligence': data.get('insights', {}).get('decision_intelligence', False),
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
        
        max_workers = min(os.cpu_count() or 1, len(md_files))
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
        system_prompt = self.config.trend_analysis_prompt
        
        user_prompt = f"Analyze this conversation and return ONLY a JSON object according to the specified format:\n\n{text}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.trend_analysis_model,
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
                    
            print(f"\nAPI Response for {filename}:\n{result[:100]}...")
            
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
                        'ai_partnership': False,
                        'ai_as_critic': False,
                        'decision_intelligence': False
                    }
                }
            else:
                # Try to parse as JSON
                try:
                    # First try direct parsing
                    analysis = json.loads(result)
                except json.JSONDecodeError as je:
                    # If that fails, try to fix common issues with single quotes
                    try:
                        print(f"\nAttempting to fix JSON with single quotes in {filename}")
                        # Replace single quotes with double quotes, but be careful with nested quotes
                        import re
                        # This is a simplified approach - might not work for all cases
                        fixed_result = result.replace("'", "\"")
                        analysis = json.loads(fixed_result)
                        print("Successfully fixed and parsed JSON with single quotes")
                    except Exception as fix_error:
                        print(f"\nJSON parsing error in {filename}. Response was:\n{result[:200]}...")
                        raise je
            
            # Save the analysis
            json_filename = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}.json")
            with open(json_filename, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            return analysis
        except json.JSONDecodeError as je:
            error_msg = f"JSON parsing error for {filename}: {str(je)}"
            print(f"\n{error_msg}")
            default_analysis = {
                'loop_completion': {'completed': False, 'exit_at_step_one': True, 'skipped_validation': False},
                'breakdown': {'exit_step': 'unknown', 'failure_reason': f'JSON parsing error: {str(je)}'},
                'insights': {'novel_patterns': False, 'ai_partnership': False, 'ai_as_critic': False, 'decision_intelligence': False}
            }
        except Exception as e:
            error_msg = f"Error analyzing {filename}: {type(e).__name__}: {str(e)}"
            print(f"\n{error_msg}")
            default_analysis = {
                'loop_completion': {'completed': False, 'exit_at_step_one': True, 'skipped_validation': False},
                'breakdown': {'exit_step': 'unknown', 'failure_reason': f'Analysis error: {type(e).__name__}: {str(e)}'},
                'insights': {'novel_patterns': False, 'ai_partnership': False, 'ai_as_critic': False, 'decision_intelligence': False}
            }
            
        # For any exception, save the default analysis
        if 'default_analysis' in locals():
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
            'ai_partnership': 1 if insights['ai_partnership'] else 0,
            'ai_as_critic': 1 if insights.get('ai_as_critic', False) else 0,
            'decision_intelligence': 1 if insights.get('decision_intelligence', False) else 0
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
        ai_as_critic = sum(1 for s in engaged_chats if s.get('ai_as_critic', False))
        decision_intelligence = sum(1 for s in engaged_chats if s.get('decision_intelligence', False))
        
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
        for s in engaged_chats:
            if s.get('completed', 0) != 1:
                exit_step = s.get('exit_step', 'unknown')
                exit_steps[exit_step] = exit_steps.get(exit_step, 0) + 1
        
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
                "Exit Steps": exit_steps
            },
            "Insights (of engaged)": {
                "Novel Patterns (%)": (novel_patterns / total_engaged) * 100,
                "AI Partnership (%)": (ai_partnership / total_engaged) * 100,
                "AI as Critic (%)": (ai_as_critic / total_engaged) * 100,
                "Decision Intelligence (%)": (decision_intelligence / total_engaged) * 100,
                "Partnership Success": {
                    "Partnerships": ai_partnership,
                    "Successful Completions with Partnership": sum(1 for s in engaged_chats 
                        if s.get('ai_partnership', False) and s.get('completed', 0) == 1),
                    "Success Rate of Partnerships (%)": (sum(1 for s in engaged_chats 
                        if s.get('ai_partnership', False) and s.get('completed', 0) == 1) / ai_partnership * 100) if ai_partnership > 0 else 0,
                    "Non-Partnership Success Rate (%)": (sum(1 for s in engaged_chats 
                        if not s.get('ai_partnership', False) and s.get('completed', 0) == 1) / (total_engaged - ai_partnership) * 100) if (total_engaged - ai_partnership) > 0 else 0
                },
                "Critical Thinking": {
                    "AI as Critic Usage": ai_as_critic,
                    "Successful Completions with AI Critic": sum(1 for s in engaged_chats 
                        if s.get('ai_as_critic', False) and s.get('completed', 0) == 1),
                    "Success Rate with AI Critic (%)": (sum(1 for s in engaged_chats 
                        if s.get('ai_as_critic', False) and s.get('completed', 0) == 1) / ai_as_critic * 100) if ai_as_critic > 0 else 0
                },
                "Decision Making": {
                    "AI-Driven Decisions": decision_intelligence,
                    "Successful Completions with AI-Driven Decisions": sum(1 for s in engaged_chats 
                        if s.get('decision_intelligence', False) and s.get('completed', 0) == 1),
                    "Success Rate with AI-Driven Decisions (%)": (sum(1 for s in engaged_chats 
                        if s.get('decision_intelligence', False) and s.get('completed', 0) == 1) / decision_intelligence * 100) if decision_intelligence > 0 else 0
                }
            }
        }
