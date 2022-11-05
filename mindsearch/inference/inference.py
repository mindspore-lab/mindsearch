# coding=utf-8
import numpy as np
from numpy import ndarray
from typing import List, Union
import mindspore.dataset.transforms as C
import mindspore.common.dtype as mstype
from mindspore.common import mutable
import mindspore.dataset as ds
from mindspore import Tensor

from mindsearch.bi_encoder.modeling import BiEncoderModel
from mindsearch.bi_encoder.base_model import BertConfig
from mindsearch.data_processor.tokenizer import FullTokenizer
from mindsearch.data_loader.data_loader import BatchExampleDataset


class BaseBiEncoderInference:
    def __init__(self, ckpt_path, config_file=None, vocab_file=None):
        if config_file is not None:
            self.bert_config = BertConfig.from_json_file(config_file)
        else:
            self.bert_config = BertConfig()
            
        self.model = BiEncoderModel.load(
            ckpt_path,
            is_training=False,
            config=self.bert_config,
            normlized=True,
            sentence_pooling_method="cls"
        )
        self.tokenizer = FullTokenizer(vocab_file, do_lower_case=True)
    
    def query_single_encode(self, query: str):
        query_dict = self.build_model_input(query)
        embedding = self.model(query=mutable(query_dict))
        return embedding.q_reps
    
    def build_model_input(self, sentence: str):
        tokens = self.tokenizer.tokenize(sentence)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        input_dict = {"input_ids": Tensor([input_ids]), "input_mask": Tensor([input_mask])}
        return input_dict
    
    def query_batch_encode(self, queries: List, batch_size: int, use_tokenizer: bool = True,
                           convert_to_numpy: bool = True) -> Union[List[Tensor], ndarray]:
        query_dataset = self.build_batch_dataset(queries, batch_size, use_tokenizer)
        
        reps = []
        for data in query_dataset.create_dict_iterator():
            inputs_ids = data["input_ids"]
            inputs = {"query": {"input_ids": inputs_ids, "input_mask": data["input_mask"]}}
            pred_scores = self.model(**inputs)
            reps.extend(pred_scores.q_reps)
        
        if convert_to_numpy:
            reps = np.asarray([Tensor.asnumpy(rep) for rep in reps])
        
        return reps
    
    def passage_batch_encode(self, passages: List, batch_size: int, use_tokenizer: bool = True,
                             convert_to_numpy: bool = True) -> Union[List[Tensor], ndarray]:
        passage_dataset = self.build_batch_dataset(passages, batch_size, use_tokenizer)
        
        reps = []
        for data in passage_dataset.create_dict_iterator():
            inputs_ids = data["input_ids"]
            inputs = {"passage": {"input_ids": inputs_ids, "input_mask": data["input_mask"]}}
            pred_scores = self.model(**inputs)
            reps.extend(pred_scores.p_reps)
        
        if convert_to_numpy:
            reps = np.asarray([Tensor.asnumpy(rep) for rep in reps])
        
        return reps
    
    def build_batch_dataset(self, sentences: List, batch_size: int, use_tokenizer: bool):
        batch_dataset = BatchExampleDataset(sentences, tokenizer=self.tokenizer, use_tokenizer=use_tokenizer)
        ms_dataset = ds.GeneratorDataset(batch_dataset, ["input_ids", "input_mask"], shuffle=False)
        
        type_cast_op = C.TypeCast(mstype.int64)
        ms_dataset = ms_dataset.map(input_columns="input_ids", operations=type_cast_op)
        ms_dataset = ms_dataset.map(input_columns="input_mask", operations=type_cast_op)
        ms_dataset = ms_dataset.batch(batch_size=batch_size, drop_remainder=False)
        return ms_dataset
