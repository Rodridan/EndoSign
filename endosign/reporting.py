"""
Automated report generation for EndoSign analysis.
"""
            
import os
import pandas as pd
import datetime

def generate_text_report(
    output_path: str,
    title: str,
    summary_dict: dict,
    biomarker_stats: pd.DataFrame = None,
    candidate_biomarkers: list = None,
    notes: str = ""
):
    """
    Generates a plain text report with project results and saves to file.
    """
    with open(output_path, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n")
        f.write(f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n")
        
        for section, value in summary_dict.items():
            f.write(f"{section}:\n")
            f.write(f"{value}\n\n")
        
        if biomarker_stats is not None:
            f.write("Top Biomarker Statistics:\n")
            f.write(biomarker_stats.head(10).to_string())
            f.write("\n\n")
        
        if candidate_biomarkers is not None:
            f.write("Candidate Biomarkers:\n")
            f.write(", ".join(candidate_biomarkers) + "\n\n")
        
        if notes:
            f.write("Notes:\n")
            f.write(notes + "\n")

def generate_markdown_report(
    output_path: str,
    title: str,
    summary_dict: dict,
    biomarker_stats: pd.DataFrame = None,
    candidate_biomarkers: list = None,
    notes: str = ""
):
    """
    Generates a Markdown report with project results and saves to file.
    """
    with open(output_path, "w") as f:
        f.write(f"# {title}\n")
        f.write(f"_Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}_\n\n")
        
        for section, value in summary_dict.items():
            f.write(f"## {section}\n")
            f.write(f"{value}\n\n")
        
        if biomarker_stats is not None:
            f.write("## Top Biomarker Statistics\n")
            f.write(biomarker_stats.head(10).to_markdown())
            f.write("\n\n")
        
        if candidate_biomarkers is not None:
            f.write("## Candidate Biomarkers\n")
            f.write(", ".join(candidate_biomarkers) + "\n\n")
        
        if notes:
            f.write("## Notes\n")
            f.write(notes + "\n")

def generate_profiling_report(df, output_path="outputs/profiling_report.html", title="EndoSign Data Profiling", explorative=True):
    """
    Generate a ydata-profiling (pandas profiling) HTML report for a DataFrame.
    """
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        raise ImportError("ydata-profiling must be installed. Install with: pip install ydata-profiling")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    profile = ProfileReport(df, title=title, explorative=explorative)
    profile.to_file(output_path)


