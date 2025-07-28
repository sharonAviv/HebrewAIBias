# example code for extracting 

# pip install pyreadstat pandas     
import pandas as pd
df, meta = pd.read_spss("ATP W41.sav", usecols=["HAPPEN2a", "WEIGHT"], convert_categoricals=True, iterator=False)

# Apply weights and get percentages
w = df["WEIGHT"]
freq = (df["HAPPEN2a"] # HAPPEN2a is a question variable name
        .value_counts(normalize=False, dropna=True)
        .to_frame("N_unweighted"))

pct = (df.groupby("HAPPEN2a")["WEIGHT"].sum() / w.sum() * 100).round(1)
freq["Pct_weighted"] = pct
print(freq)

