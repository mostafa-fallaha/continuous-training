# continuous-training

All the scripts contains comments to explain what im doing

# How to use this

**1. in the CLI, run `python .\generate_data.py 3000`.** <br>
this is just to simulate a data stream, every time you specify a number higher than the last one.
<br><br>

**2. Also in the CLI, run `run_versioning.ps1 version_new_data.py "Add 3k"`** <br>
now after i generated some data, i want to push it to my Storage (Google Drive) and version it.<br>
this command and the one bedore are both doing the simulation of adding new data.

# Explaining each file

### 1. data_cleaning.py:

this script it's just cleaning the data, the user needs to write this based on his dataset.

### 2. initial_model_training.py:

the user needs to write the code of his model here, it needs to be an online machine learning model.<br>
it gets the data for the model from 'data_cleaning.py'.

### 3.model_trianing.py:

this script it's not done by the user. it's just loading the model created by the user earlier.<br>
and loading the data from 'data_cleaning.py' and taking the new added data only.<br>
finally the model is learning from the new added data.

### 4.generate_data.py:

this takes a number, and generate data with number of rows equal to the passed number.<br>
if 3000 was passed -> it will generate a dataset with 3000 rows.<br>
this is just to simulate having a source that is streaming some data.

### 5.check_drifting.py:

this script will check if there's any drifting, it will comapre each column in the newly generated data (df_cur) to it's corresponding one in the data on the remote storage (df_ref). and return 0 or 1. (0 = no drift | 1 = drift).

### 6.version_new_data.py

this script will version and push the newly generated data. but before, it will check for drifting by calling 'check_drifting.py'. <br>
if yes, it will trigger the model to start the learning proccess 'model_training.py', and no otherwise.
and it will version and push the new data anyway.
