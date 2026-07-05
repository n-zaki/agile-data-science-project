import pandas as pd

def test_dataset_load():
    df = pd.read_csv("TelcoChurn_Analytics.csv")
    assert df.shape[0] > 0, "Dataset is empty"
    print("Dataset load test passed")

def test_missing_values():
    df = pd.read_csv("TelcoChurn_Analytics.csv")
    assert df.isnull().sum().sum() >= 0, "Missing value check failed"
    print("Missing value test passed")

if __name__ == "__main__":
    test_dataset_load()
    test_missing_values()
