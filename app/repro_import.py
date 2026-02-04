import os
import sys

# Simulate the path fix in main.py
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

print(f"CWD: {os.getcwd()}")
print(f"Added to path: {root_dir}")
print(f"Sys Path: {sys.path}")

try:
    import src
    print("Import src SUCCESS")
    from src.pipeline import detector
    print("Import src.pipeline.detector SUCCESS")
except ImportError as e:
    print(f"Import FAILED: {e}")
