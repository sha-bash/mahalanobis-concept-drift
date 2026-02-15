#!/usr/bin/env python3
"""
Demo script for Mahalanobis Concept Drift Detector.

Runs evaluation on the Kaggle multilingual customer support tickets dataset.
"""

import subprocess
import sys
import os

def main():
    # Path to archive
    archive_path = os.path.join('data', 'archive.zip')
    if not os.path.exists(archive_path):
        print(f"Error: Archive not found at {archive_path}")
        sys.exit(1)

    # Run eval with auto-demo
    cmd = [
        sys.executable, '-m', 'mcd.cli', 'eval',
        '--data', archive_path,
        '--auto-demo'
    ]

    print("Running demo evaluation...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running evaluation:")
        print(result.stderr)
        sys.exit(1)

    # Print summary
    print("Demo evaluation completed successfully!")
    print("Outputs saved to reports/demo_run")
    print("\nSummary from README:")
    
    readme_path = os.path.join('reports', 'demo_run', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            print(f.read())
    else:
        print("README not found.")

if __name__ == '__main__':
    main()