from sporc import SPORCDataset

ds = SPORCDataset("/shared/6/projects/sporc/v1")
print(ds.get_dataset_statistics())