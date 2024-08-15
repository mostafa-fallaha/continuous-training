import pandas as pd
from pathlib import Path
import argparse

# Getting the nb of rows from the cmd
parser = argparse.ArgumentParser(description="nb of rows in the data")
parser.add_argument("nb_of_rows", type=int, help="specify nb of rows")
args = parser.parse_args()
nb_of_rows = args.nb_of_rows

# Let's say this csv is my source of data
df = pd.read_csv("data/Google-Playstore.csv")
print(df.shape)

# every time im taking a 1000 rows more than the last one
# Like it's streaming 1000 new rows to me every minute for example
df = df.iloc[:nb_of_rows]

print(df.shape)

# saving to push it with 'version_new_data.py'
filepath = Path('data/new_data.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(filepath, index=False)
