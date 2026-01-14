
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
print("your torch library in here:{}".format(torch.__path__))
print("your nn library in here:{}".format(nn.__path__))
print("your optim library in here:{}".format(optim.__path__))

# print(type(DataLoader), type(nn))
# print("your DataLoader library in here:{}".format(DataLoader.__path__))  # error
# print("your Dataset library in here:{}".format(Dataset.__path__))        # error