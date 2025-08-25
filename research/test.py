import pandas as pd, sys, glob
for p in glob.glob(r"..\\frontend\\public\\*.csv"):
    df = pd.read_csv(p, nrows=5)
    print(p, "→ cols:", list(df.columns))