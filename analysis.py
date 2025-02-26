import os
import re
import sys

def analyze_md_files(directory):
    # Regular expression pattern to identify Loop Completion Analysis sections
    completion_pattern = re.compile(
        r"### 4\.1 Loop Completion Analysis\n- \*\*Did the USER complete all five steps of the AI Decision Loop\?\*\*\n\s*-\s*(Yes|No)", 
        re.IGNORECASE
    )

    # Initialize counters
    total_chats = 0
    completed_loops = 0
    not_completed_loops = 0

    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Scan all markdown files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            file_path = os.path.join(directory, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                md_text = file.read()

            # Extract "Loop Completion Analysis" sections
            loop_completion_sections = completion_pattern.findall(md_text)

            # Update global counters
            total_chats += len(loop_completion_sections)
            completed_loops += sum(1 for result in loop_completion_sections if result.lower() == "yes")
            not_completed_loops += sum(1 for result in loop_completion_sections if result.lower() == "no")

    # Calculate percentages
    completed_percentage = (completed_loops / total_chats) * 100 if total_chats > 0 else 0
    not_completed_percentage = (not_completed_loops / total_chats) * 100 if total_chats > 0 else 0

    # Display summary results
    summary = {
        "Total Chats Analyzed": total_chats,
        "Completed Loop (%)": completed_percentage,
        "Not Completed Loop (%)": not_completed_percentage,
    }

    print("\nOverall Summary:")
    print(summary)

# Run the script with a directory path passed as an argument
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_md_files.py /path/to/markdown/files")
    else:
        directory_path = sys.argv[1]
        analyze_md_files(directory_path)
