import pandas as pd
import numpy as np

# Load the existing dataset
df = pd.read_csv("../placementdata.csv")

# Set seed for reproducibility
np.random.seed(42)

# Generate CommunicationScore (realistic distribution)
communication_scores = np.random.normal(loc=70, scale=12, size=10000)
communication_scores = np.clip(communication_scores, 0, 100).astype(int)

# Add the new column
df["CommunicationScore"] = communication_scores

# Save the updated dataset
df.to_csv("placementdata.csv", index=False)

print("CommunicationScore column added successfully!")
print(df.head())