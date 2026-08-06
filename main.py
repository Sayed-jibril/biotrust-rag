# main.py
# -*- coding: utf-8 -*-

from pipeline import run_pipeline
import os
import sys
# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(__file__))


if __name__ == "__main__":
    # Optional: you can set environment variables or configure here
    # Then run the full pipeline
    run_pipeline()
