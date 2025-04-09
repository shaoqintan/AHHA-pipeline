import subprocess
import sys
import os
import numpy as np
import pickle

# File paths (update these with your actual paths)
C_SHARP_EXECUTABLE = "path/to/your/C#/executable.exe"
ACCELERATION_DATA_FILE = "acceleration_data.txt"  # Changed to .txt
LINEAR_MOVEMENT_OUTPUT = "linear_movements.npy"
FEATURE_EXTRACTION_OUTPUT = "features.npy"
SYSTEM_VARIANCE_FILE = "system_variance.npy"
MODEL_FILE = "model.pkl"
FINAL_SCORE_FILE = "score.txt"

def run_pipeline(id, week):
    try:
        # --------------------------------------------------------------------------
        # Step 1: Run C# code to generate acceleration data (with 10-second timeout)
        # --------------------------------------------------------------------------
        print("Step 1/4: Running C# code to collect accelerometer data...")
        try:
            csharp_process = subprocess.run(
                [C_SHARP_EXECUTABLE],
                capture_output=True,
                text=True,
                timeout=10  # 10-second timeout
            )
        except subprocess.TimeoutExpired:
            print("C# process exceeded 10-second timeout!")
            sys.exit(1)

        if csharp_process.returncode != 0:
            print("C# process failed!")
            print("Error:", csharp_process.stderr)
            sys.exit(1)

        print("C# output:", csharp_process.stdout)
        print("Acceleration data written to:", ACCELERATION_DATA_FILE)

        # --------------------------------------------------------------------------
        # Step 2: Run linear movement detection (ID/week version)
        # --------------------------------------------------------------------------
        print("\nStep 2/4: Detecting linear movements...")
        linear_movement_process = subprocess.run(
            [
                "python",
                "linear_movement_detection.py",
                f"{id}_{week}"  # Pass ID and week as arguments
            ],
            capture_output=True,
            text=True
        )

        if linear_movement_process.returncode != 0:
            print("Linear movement detection failed!")
            print("Error:", linear_movement_process.stderr)
            sys.exit(1)

        print("Linear movements saved to:", LINEAR_MOVEMENT_OUTPUT)

        # --------------------------------------------------------------------------
        # Step 3: Run feature extraction (ID/week version)
        # --------------------------------------------------------------------------
        print("\nStep 3/4: Extracting features...")
        feature_extraction_process = subprocess.run(
            [
                "python",
                "feature_extraction.py",
                f"{id}_{week}"  # Pass ID and week as arguments
            ],
            capture_output=True,
            text=True
        )

        if feature_extraction_process.returncode != 0:
            print("Feature extraction failed!")
            print("Error:", feature_extraction_process.stderr)
            sys.exit(1)

        print("Features saved to:", FEATURE_EXTRACTION_OUTPUT)

        # --------------------------------------------------------------------------
        # Step 4: Run model prediction
        # --------------------------------------------------------------------------
        print("\nStep 4/4: Running model prediction...")

        # Load the model
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)

        # Load features
        features = np.load(FEATURE_EXTRACTION_OUTPUT)

        # Get prediction
        score = model.predict(features.reshape(1, -1))

        # Save score
        np.savetxt(FINAL_SCORE_FILE, score)

        print("\nFinal Score:", score[0])
        print("Pipeline completed successfully!")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Check for ID and week arguments
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py <ID> <Week>")
        sys.exit(1)

    # Parse ID and week from command line
    id = sys.argv[1]
    week = sys.argv[2]

    # Cleanup previous files
    for f in [ACCELERATION_DATA_FILE, LINEAR_MOVEMENT_OUTPUT,
              FEATURE_EXTRACTION_OUTPUT, FINAL_SCORE_FILE]:
        if os.path.exists(f):
            os.remove(f)

    # Run the pipeline
    run_pipeline(id, week)