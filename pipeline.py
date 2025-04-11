import subprocess
import sys
import os
import signal
import pickle
import numpy as np

# File paths (update with your paths)
C_SHARP_EXECUTABLE = r"C:\AHHA Lab\Shimmer\ShimmerConsoleAppExample\ShimmerConsoleAppExample\bin\Debug\ShimmerConsoleAppExample.exe"
FEATURE_EXTRACTION_SCRIPT = r"C:\AHHA Lab\AHHA Pipeline\feature_extraction.py"

def run_pipeline():
    try:
        # Step 1: Start C# process
        # print("Step 1: Running C# data collection...")
        # csharp_process = subprocess.Popen(
        #     [C_SHARP_EXECUTABLE],
        #     creationflags=subprocess.CREATE_NEW_CONSOLE
        # )

        # print("C# process started. Press ENTER to stop it and continue with feature extraction.")
        # input()  # Wait for user input to stop the C# process

        # # Terminate C# process
        # csharp_process.terminate()
        # try:
        #     csharp_process.wait(timeout=5)
        # except subprocess.TimeoutExpired:
        #     csharp_process.kill()

        # stdout, stderr = csharp_process.communicate()
        # print("C# process output:\n", stdout)
        # if stderr:
        #     print("C# process error output:\n", stderr)

        # Step 2: Run feature extraction script
        print("\nStep 2: Running feature extraction...")
        feature_process = subprocess.run(
            ["python", FEATURE_EXTRACTION_SCRIPT],
            capture_output=True,
            text=True
        )

        if feature_process.returncode != 0:
            print("Feature extraction failed!")
            print("Error:", feature_process.stderr)
            sys.exit(1)

        print("Feature extraction output:")
        print(feature_process.stdout.strip())

        # features_array = np.load("features/features_array.npy")

        # Load model
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        # Predict
        predictions = model.predict(features_array)
        print("Predicted mobility scores:")
        print(predictions)

        print("\nPipeline completed successfully!")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
