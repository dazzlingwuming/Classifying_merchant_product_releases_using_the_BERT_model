import sys

from preprocess.data_load import get_dataloader, DatasetType
from runner.evaluate import run_evaluate
from runner.predict import predict
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

def f3():
    run_evaluate()

def f4(text): #运行不成功，因为没有加载模型
    predict(text)

if __name__ == "__main__":
    # load_data_processor()
    # print(sys.path)
    # f1()
    # f2()
    # f3()
    # f4("蒙牛牛奶300g草莓味冰淇淋")


    pass