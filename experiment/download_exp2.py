from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("eth-sri/agentbench")


def save_split(split, csv_path, json_path):
    df = split.to_pandas()
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records")
    print(f"Saved {csv_path} and {json_path}")

if hasattr(ds, "items"):
    for split_name, split in ds.items():
        csv_path = f"agentbench_{split_name}.csv"
        json_path = f"agentbench_{split_name}.json"
        save_split(split, csv_path, json_path)
else:
    save_split(ds, "agentbench.csv", "agentbench.json")