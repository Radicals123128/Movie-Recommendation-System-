import pandas as pd
df = pd.read_csv('compressed_data.csv', compression='zip')


# Check existing columns
print(df.columns)

# Use 'description' instead of 'genres'
df["description"] = df["description"].str.lower().str.replace(",", " ")
