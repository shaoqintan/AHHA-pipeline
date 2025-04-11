import subprocess
import sys
import os
import signal
import pickle
from time import sleep
import numpy as np
from feature_extraction import linear_movements_detection, load

# File paths (update with your paths)
C_SHARP_EXECUTABLE = r"C:\AHHA Lab\Shimmer\ShimmerConsoleAppExample\ShimmerConsoleAppExample\bin\Debug\ShimmerConsoleAppExample.exe"
FEATURE_EXTRACTION_SCRIPT = r"C:\AHHA Lab\AHHA Pipeline\feature_extraction.py"

def run_pipeline():
    try:
        # Step 1: Start C# process
        print("Step 1: Running C# data collection...")
        csharp_process = subprocess.Popen(
            [C_SHARP_EXECUTABLE],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )

        print("C# process started. Press ENTER to stop it and continue with feature extraction.")
        input()  # Wait for user input to stop the C# process

        # Terminate C# process
        sleep(3)
        csharp_process.terminate()
        try:
            csharp_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            csharp_process.kill()

        stdout, stderr = csharp_process.communicate()
        print("C# process output:\n", stdout)
        if stderr:
            print("C# process error output:\n", stderr)
        
        # Step 2: Run feature extraction script
        print("\nStep 2: Running feature extraction...")
        start_idx, end_idx = linear_movements_detection(1, "Test")
        features_array = load(start_idx, end_idx, split='')

        # Step 3: Run model prediction
        model_path = "model.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        prediction = model.predict(features_array)
        print("Mobility prediction:", prediction)

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
