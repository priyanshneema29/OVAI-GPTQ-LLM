import os
import requests
import pandas as pd

# ref: https://huggingface.co/datasets/lonestar108/enlightenedllm
dataset_name = "lonestar108/enlightenedllm"
API_URL = f"https://datasets-server.huggingface.co/parquet?dataset={dataset_name}"
def get_dataset_files():
    response = requests.get(API_URL)
    data = response.json()["parquet_files"]
    
    dataset = {}
    for file in data:
        dataset[file["split"]] = file["url"]
    return dataset 

def load_dataset(path, split, save=False):
    data = pd.read_parquet(path)   
    if(save):
        data.to_csv(os.path.join("data", f"{dataset_name.replace('/', '_')}_{split}.csv"),
                    index=False)
    return data

if __name__ == "__main__":
    # Get dataset from hugging face hub
    dataset = get_dataset_files()
    for split, path in dataset.items():
        print(f"Downloading {split} from {path}")
        load_dataset(path, split, save=True)

    # Test hugging face dataset
    from datasets import Dataset
    data_path = "data"
    dataset_name = "lonestar108_enlightenedllm"
    train_data = Dataset.from_csv(os.path.join(data_path, f"{dataset_name}_train.csv"))
    validation_data = Dataset.from_csv(os.path.join(data_path, f"{dataset_name}_validation.csv"))
    test_data = Dataset.from_csv(os.path.join(data_path, f"{dataset_name}_test.csv"))

    print(train_data)