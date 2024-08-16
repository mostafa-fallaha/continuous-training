# continuous-training

All the scripts contains comments to explain what im doing

# How to use this

**1. in the CLI, run `python .\generate_data.py 3000`.** <br>
you can start by 1000, and increase it every time by 1000
<br><br>

**2. Also in the CLI, run `run_versioning.ps1 version_new_data.py "Add 3k"`** <br>
Change the commit message to specify the nb of added rows

# Explaining each file

### 1.model_trianing.py:

this script is just training the model on the latest data (versioned with DVC) on the Storage (google drive).

### 2.generate_data.py:

this takes a number, and generate data with number of rows equal to the passed number.<br>
if 3000 was passed -> it will generate a dataset with 3000 rows.<br>
this is just to simulate having a source that is streaming some data.

### 3.check_drifting.py:

this script will check if there's any drifting, it will comapre the newly generated data (df_cur) to the data on the remote storage (df_ref). and return 0 or 1. (0 = no drift | 1 = drift).

### 4.version_new_data.py

this script will version and push the newly generated data. but before, it will check for drifting by calling 'check_drifting.py'. <br>
if yes, it will trigger a retrain for the model, and no otherwise.
and it will version and push the new data anyway.
