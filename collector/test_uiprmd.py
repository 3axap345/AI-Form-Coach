from aeon.datasets import load_rehab_pile_dataset

X_train, y_train = load_rehab_pile_dataset(
    "UIPRMD-DS-C",
    split="train"
)

X_test, y_test = load_rehab_pile_dataset(
    "UIPRMD-DS-C",
    split="test"
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)
print("Labels:", y_train)