import subprocess
import argparse
import check_drifting

# Getting the commit message from the cmd
parser = argparse.ArgumentParser(description="Run DVC and Git commands.")
parser.add_argument("commit_message", type=str, help="Commit message for Git")
args = parser.parse_args()

# Define files paths
data_file = "data/new_data.csv"
dvc_file = "data/new_data.csv.dvc"
gitignore_file = "data/.gitignore"

commit_message = args.commit_message

results = check_drifting.check_drift_all_features()

# Run DVC and Git commands
subprocess.run(["dvc", "add", data_file], check=True)
subprocess.run(["git", "add", dvc_file, gitignore_file, 'run_versioning.ps1', 'version_new_data.py',
                'generate_data.py', 'model_training.py', 'README.md', 'images/', '.dvc/config'], check=True)

subprocess.run(["git", "commit", "-m", commit_message], check=True)
subprocess.run(["dvc", "push"], check=True)
subprocess.run(["git", "push"], check=True)

# If everything above worked without issues, the training will start
# I will try to figure out a way better than this

if results == 1:
    print("\n============ Data drift detected, Training triggered ===================\n")
    subprocess.run(["python", "model_training.py"], check=True)
else:
    print("\n=================== No Data drift ===================\n")