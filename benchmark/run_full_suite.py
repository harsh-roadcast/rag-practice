import subprocess
import sys
import os

MODES = [
    ("recursive", "small"),
    ("recursive", "medium"),
    ("recursive", "large"),
    ("length", "small"),
    ("length", "medium"),
    ("length", "large")
]

PYTHON_EXE = sys.executable
SCRIPT_PATH = "benchmark/run_single.py"

print("🔥 STARTING FULL BENCHMARK SUITE 🔥")
print(f"Executing with: {PYTHON_EXE}")

for strategy, size in MODES:
    print(f"\n==================================================")
    print(f"▶️  STARTING: Strategy={strategy}, Size={size}")
    print(f"==================================================")
    
    try:
        # Run as a separate process to ensure memory is cleared after each run
        result = subprocess.run(
            [PYTHON_EXE, SCRIPT_PATH, "--strategy", strategy, "--size", size],
            check=True,
            capture_output=False  # Let stdout flow to terminal so user sees progress
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {strategy}-{size}: {e}")
        # We continue to the next one even if one fails
        continue

print("\n✅ FULL SUITE COMPLETE!")
print("Check benchmark/results/final_benchmark_results.csv for data.")
