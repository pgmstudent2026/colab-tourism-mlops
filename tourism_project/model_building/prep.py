import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")  # registered tourism.csv inside the data folder

# Drop columns that are not predictive features:
# - CustomerID is a unique identifier
# - Unnamed: 0 is a leftover index column from a previous export and carries no information
drop_cols = [c for c in ["CustomerID", "Unnamed: 0"] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

# Fix a known data-entry typo in Gender ("Fe Male" -> "Female"). This is a
# deterministic value correction (not derived from the target), so it is safe
# to do before the train/test split.
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

target = "ProdTaken"  # the column to predict: 1 if the customer purchased the package, else 0
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # stratify on the target so both splits keep the same purchase ratio
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
