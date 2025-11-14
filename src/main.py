import sys

from preprocess.data_load import get_dataloader, DatasetType
from runner.train import train

# from transformers import DataProcessor
# from preprocess.data_preprocess import load_data_processor
# from runner.train import train


def f1():
    train_dataloader = get_dataloader(DatasetType.Train)
    test_dataloader = get_dataloader(DatasetType.Test)
    print(train_dataloader)
    print(test_dataloader)
    for batch in train_dataloader:
        print(batch.keys())
        break

def f2():
    train()

if __name__ == "__main__":
    # load_data_processor()
    # print(sys.path)
    # f1()
    f2()