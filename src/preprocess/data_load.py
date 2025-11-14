from enum import Enum

from datasets import load_from_disk
from torch.utils.data import DataLoader

from config import configs


class DatasetType(Enum):
    Train = "train"
    Validation = "validation"
    Test = "test"
    '''
    为什么使用枚举类型？
    使用枚举类型可以提高代码的可读性和可维护性。通过定义枚举类型，可以清晰地表示数据集的不同类型，避免使用字符串常量时可能出现的拼写错误或不一致问题。
    使用枚举类型还可以方便地进行类型检查，确保传递给函数的参数是有效的枚举成员。
    并且强制使用预定义的选项，避免了无效输入，提高了代码的健壮性。
    '''

def get_dataloader(data_type = DatasetType.Train):
    path = str(configs.PROCESSED_DIR/data_type.value)
    '''
    data_type.value 获取枚举成员的值，这里返回的是字符串 "train"、"validation" 或 "test"。
    但是data_type是DatasetType类型，不能直接与Path对象进行拼接，所以需要使用data_type.value来获取字符串值。
    '''

    data_train = load_from_disk(path)
    data_train.set_format(type = "torch")#设置数据集的格式为 PyTorch 张量格式，以便后续在模型训练中直接使用。

    Dataload = DataLoader(data_train , batch_size= configs.BATCH_SIZE, shuffle=True)

    return Dataload

if __name__ == "__main__":
    dataloader = get_dataloader(DatasetType.Train)
    for batch in dataloader:
        print(batch)
        break