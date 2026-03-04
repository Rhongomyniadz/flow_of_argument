from sporc import SPORCDataset

ds = SPORCDataset(parquet_dir="/shared/6/projects/sporc/v1")
print(ds.get_dataset_statistics())