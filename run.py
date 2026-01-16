#!/usr/bin/env python
"""
Runner script for doc-bases project.
This script properly handles Python module imports when running from the project root.
"""
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import and run the main module
from src.main import main

if __name__ == "__main__":
    main()
