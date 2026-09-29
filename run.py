"""
run.py
───────
Convenience launcher for the AutoReview Streamlit application.

Usage:
    python run.py

Or directly:
    streamlit run app/main.py
"""

import subprocess
import sys
import os


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    main_app = os.path.join(project_root, "app", "main.py")

    # Use the .venv Python if available (works even when python is not in PATH)
    venv_streamlit = os.path.join(project_root, ".venv", "Scripts", "streamlit.exe")
    if os.path.exists(venv_streamlit):
        cmd = [
            venv_streamlit, "run", main_app,
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
        ]
    else:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            main_app,
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
        ]

    print("🚗 Starting AutoReview — Automotive Sentiment Analytics")
    print(f"   App: {main_app}")
    print("   URL: http://localhost:8501\n")

    subprocess.run(cmd, cwd=project_root)


if __name__ == "__main__":
    main()
