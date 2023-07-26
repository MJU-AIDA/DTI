import pandas as pd

# Load the vec_dev_origin.csv, vec_train_origin.csv, and vec_test_origin.csv files into dataframes
dev_df = pd.read_csv("vec_dev_origin.csv")
train_df = pd.read_csv("vec_train_origin.csv")
test_df = pd.read_csv("vec_test_origin.csv")

# Extract the gene_enco, drug_enco, and Y columns from each dataframe
dev_df = dev_df[["gene_enco", "drug_enco", "Y"]]
train_df = train_df[["gene_enco", "drug_enco", "Y"]]
test_df = test_df[["gene_enco", "drug_enco", "Y"]]

# Convert all values in the dataframes to integers
dev_df = dev_df.astype(int)
train_df = train_df.astype(int)
test_df = test_df.astype(int)

# Convert the dataframes to tab-separated strings with space-separated columns
dev_str = dev_df.to_csv(sep=" ", index=False, header=False)
train_str = train_df.to_csv(sep=" ", index=False, header=False)
test_str = test_df.to_csv(sep=" ", index=False, header=False)

# Save the resulting strings to text files
with open("dev.txt", "w") as f:
    f.write(dev_str)
with open("train.txt", "w") as f:
    f.write(train_str)
with open("test.txt", "w") as f:
    f.write(test_str)
