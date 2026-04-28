import pandas as pd

# Correct dataset path
file_path = "data/raw/application_train.csv"

# Load dataset
df = pd.read_csv(file_path)

print("Dataset loaded successfully ✅")
print("Shape:", df.shape)

print("\nFirst 5 columns:")
print(df.columns[:5])