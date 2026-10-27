"""
Top-level entry script to launch DLP4710 3D Measurement Software.
Usage:
  python run_measurement.py
"""

import importlib
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def main():
    meas_app = importlib.import_module("SL-3D-Measurement-DLP4710-SingleCamera.app")
    meas_app.main()


if __name__ == "__main__":
    main()

