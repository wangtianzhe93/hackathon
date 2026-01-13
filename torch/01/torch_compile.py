# Author: Tianzhe

from transformers import BertModel, BertTokenizer
from typing import List, Optional
import time
import torch
from torchvision import models
import numpy as np

COMPILE_MODE = "default reduce-overhead max-autotune"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sin_func(x):
    return torch.sin(x) + torch.cos(x)


def test_sin_func(
    x: Optional[torch.Tensor] = None,
    mode_list: Optional[List[str]] = ["default"],
    num_iters: int = 1000
):
    assert x is not None
    assert mode_list is not None
    for mode in mode_list:
        module_compiled = torch.compile(sin_func, mode=mode)
        sin_func(x)
        module_compiled(x)

def test_resnet(
    mode_list: Optional[List[str]] = ["default"],
    num_iters: int = 1000
):
    resnet18 = models.resnet18().to(device)
    resnet18.eval() # evaluation mode
    fake_image = torch.randn(16, 3, 224, 224).to(device) # 16 fake images
    with torch.no_grad():
        module_compiled = torch.compile(resnet18)
        # warm up
        resnet18(fake_image)
        module_compiled(fake_image)

        for i in range(num_iters):
            resnet18(fake_image)
        
        for i in range(num_iters):
            module_compiled(fake_image)


def test_bert():
    bert = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    input_text = "Here is some text to encode"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    bert.to(device)
    bert.eval()

    num_iters = 2
    with torch.no_grad():
        bert_compiled = torch.compile(bert)
        # warm up
        bert(**inputs)
        bert_compiled(**inputs)

        for _ in range(num_iters): 
            _ = bert(**inputs)

        for _ in range(num_iters):
            _ = bert_compiled(**inputs)


if __name__ == "__main__":

    fake_input = torch.Tensor(1).to(device)

    # test_sin_func_compile(x=fake_input, mode_list=COMPILE_MODE.split())
    # test_resnet(num_iters=2)
    test_bert()


