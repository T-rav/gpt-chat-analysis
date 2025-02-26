import os
import re
import sys
import pandas as pd

def analyze_md_files(directory):
    # Regular expression pattern to identify Loop Completion Analysis sections
    completion_pattern = re.compile(
        r"### 4\.1 Loop Completion Analysis\n- \*\*Did the USER complete all five steps of the AI Decision Loop\?\*\*\n\s*-\s*(Yes|No)", 
        re.IGNORECASE
    )

    results = []

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

            # Count completions and non-completions
            completed_loops = sum(1 for result in loop_completion_sections if result.lower() == "yes")
            not_completed_loops = sum(1 for result in loop_completion_sections if result.lower() == "no")
            total_chats = completed_loops + not_completed_loops

            # Append results
            results.append({
                "File": filename,
                "Total Chats": total_chats,
                "Completed Loops": completed_loops,
                "Not Completed Loops": not_completed_loops
            })

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Calculate overall totals
    total_chats = df_results["Total Chats"].sum()
    total_completed_loops = df_results["Completed Loops"].sum()
    total_not_completed_loops = df_results["Not Completed Loops"].sum()

    # Calculate percentages
    completed_percentage = (total_completed_loops / total_chats) * 100 if total_chats > 0 else 0
    not_completed_percentage = (total_not_completed_loops / total_chats) * 100 if total_chats > 0 else 0

    # Display summary results
    summary = {
        "Total Chats Analyzed": total_chats,
        "Completed Loop (%)": completed_percentage,
        "Not Completed Loop (%)": not_completed_percentage,
    }

    print("\nOverall Summary:")
    print(summary)

    # Print detailed file-wise breakdown
    if not df_results.empty:
        print("\nDetailed File Analysis:")
        print(df_results.to_string(index=False))
    else:
        print("\nNo Markdown files found or no chats detected.")

# Run the script with a directory path passed as an argument
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_md_files.py /path/to/markdown/files")
    else:
        directory_path = sys.argv[1]
        analyze_md_files(directory_path)
