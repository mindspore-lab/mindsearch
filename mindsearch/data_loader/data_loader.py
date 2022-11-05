# coding=utf-8# coding=utf-8
import json
from typing import List

from mindspore import Tensor


def read_json_lines(file):
    datas = []
    with open(file, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            obj = json.loads(line)
            datas.append(obj)
    return datas


def padding_to_max(ids: List, padding_id=0, max_len=128):
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        num_paddings = max_len - len(ids)
        ids.extend([padding_id] * num_paddings)
    return ids


class BatchExampleDataset:
    def __init__(self, examples: List, tokenizer, key_field: str = "text", max_len=128, use_tokenizer=False):
        self.datas = examples
        self.tokenizer = tokenizer
        self.key_field = key_field
        self.max_len = max_len
        self.use_tokenizer = use_tokenizer
    
    def __getitem__(self, item):
        text = self.datas[item]
        if self.use_tokenizer:
            tokens = self.tokenizer.tokenize(text)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            input_ids = text
        input_mask = [1] * len(input_ids)
        
        input_ids = Tensor(padding_to_max(input_ids))
        input_mask = Tensor(padding_to_max(input_mask))
        return input_ids, input_mask
    
    def __len__(self):
        return len(self.datas)
