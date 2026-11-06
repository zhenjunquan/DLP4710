"""
Top-level entry script to launch DLP4710 Macro Focus Assistant.
Usage:
  python run_focus.py
"""

import os
import sys

# Ensure dlp_common and project root are in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from DLP4710_Focus_Assistant.app import main

if __name__ == "__main__":
    main()

