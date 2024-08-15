import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="specify nb of rows")
parser.add_argument("nb_of_rows", type=int, help="nb of rows in the data")
args = parser.parse_args()
nb_of_rows = args.nb_of_rows

df = pd.read_csv("data/Google-Playstore.csv")
print(df.shape)

df = df.iloc[:nb_of_rows]

print(df.shape)

filepath = Path('data/new_data.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(filepath, index=False)
