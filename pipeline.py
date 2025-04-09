import subprocess
import sys
import os
import numpy as np
import pickle

# File paths (update these with your actual paths)
C_SHARP_EXECUTABLE = "path/to/your/C#/executable.exe"
ACCELERATION_DATA_FILE = "acceleration_data.csv"
LINEAR_MOVEMENT_OUTPUT = "linear_movements.npy"
FEATURE_EXTRACTION_OUTPUT = "features.npy"
SYSTEM_VARIANCE_FILE = "system_variance.npy"
MODEL_FILE = "model.pkl"
FINAL_SCORE_FILE = "score.txt"

SCRIPT_DIR = r"C:\AHHA Lab\Ryan"
FEATURE_EXTRACTION_SCRIPT = os.path.join(SCRIPT_DIR, "feature_extraction.py")

def run_pipeline():
    try:
        result = subprocess.run(
            ["python", FEATURE_EXTRACTION_SCRIPT, "1_test"],  # Pass arguments if needed
            capture_output=True,  # Capture the output
            text=True  # Ensure output is treated as text
        )

        # Print raw output (for debugging)
        # print("Raw Output:\n", result.stdout)

        # Extract the actual value (hi)
        hi_output = result.stdout.strip()  # Remove extra whitespace
        print("Extracted hi:", hi_output)
        # # --------------------------------------------------------------------------
        # # Step 1: Run C# code to generate acceleration data
        # # --------------------------------------------------------------------------
        # print("Step 1/4: Running C# code to collect accelerometer data...")
        # csharp_process = subprocess.run(
        #     [C_SHARP_EXECUTABLE],
        #     capture_output=True,
        #     text=True
        # )
        
        # if csharp_process.returncode != 0:
        #     print("C# process failed!")
        #     print("Error:", csharp_process.stderr)
        #     sys.exit(1)
            
        # print("C# output:", csharp_process.stdout)
        # print("Acceleration data written to:", ACCELERATION_DATA_FILE)

        # # --------------------------------------------------------------------------
        # # Step 2: Run linear movement detection
        # # --------------------------------------------------------------------------
        # print("\nStep 2/4: Detecting linear movements...")
        # linear_movement_process = subprocess.run(
        #     [
        #         "python", 
        #         "linear_movement_detection.py",
        #         "--input", ACCELERATION_DATA_FILE,
        #         "--output", LINEAR_MOVEMENT_OUTPUT
        #     ],
        #     capture_output=True,
        #     text=True
        # )
        
        # if linear_movement_process.returncode != 0:
        #     print("Linear movement detection failed!")
        #     print("Error:", linear_movement_process.stderr)
        #     sys.exit(1)
            
        # print("Linear movements saved to:", LINEAR_MOVEMENT_OUTPUT)

        # # --------------------------------------------------------------------------
        # # Step 3: Run feature extraction
        # # --------------------------------------------------------------------------
        # print("\nStep 3/4: Extracting features...")
        # feature_extraction_process = subprocess.run(
        #     [
        #         "python",
        #         "feature_extraction.py",
        #         "--input", LINEAR_MOVEMENT_OUTPUT,
        #         "--system_variance", SYSTEM_VARIANCE_FILE,
        #         "--output", FEATURE_EXTRACTION_OUTPUT
        #     ],
        #     capture_output=True,
        #     text=True
        # )
        
        # if feature_extraction_process.returncode != 0:
        #     print("Feature extraction failed!")
        #     print("Error:", feature_extraction_process.stderr)
        #     sys.exit(1)
            
        # print("Features saved to:", FEATURE_EXTRACTION_OUTPUT)

        # # --------------------------------------------------------------------------
        # # Step 4: Run model prediction
        # # --------------------------------------------------------------------------
        # print("\nStep 4/4: Running model prediction...")
        
        # # Load the model
        # with open(MODEL_FILE, "rb") as f:
        #     model = pickle.load(f)
            
        # # Load features
        # features = np.load(FEATURE_EXTRACTION_OUTPUT)
        
        # # Get prediction
        # score = model.predict(features.reshape(1, -1))
        
        # # Save score
        # np.savetxt(FINAL_SCORE_FILE, score)
        
        # print("\nFinal Score:", score[0])
        # print("Pipeline completed successfully!")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Cleanup previous files
    for f in [ACCELERATION_DATA_FILE, LINEAR_MOVEMENT_OUTPUT, 
              FEATURE_EXTRACTION_OUTPUT, FINAL_SCORE_FILE]:
        if os.path.exists(f):
            os.remove(f)
    
    run_pipeline()