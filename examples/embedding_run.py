# coding=utf-8
import os
import numpy as np

import mindspore
from tokenizers import BertWordPieceTokenizer
from mindsearch.inference.inference import BaseBiEncoderInference
from mindsearch.data_loader.data_loader import read_json_lines
from mindsearch.utils.logger import Logger

mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="CPU")
logger = Logger(__name__).get_logger()


def main():
    model_path = ""
    config_file = os.path.join(model_path, "config.json")
    ckpt_path = os.path.join(model_path, "mindspore.ckpt")
    vocab_path = os.path.join(model_path, "vocab.txt")
    query_data_path = ""
    prediction_save_path = ""
    sentence = "Artificial intelligence was founded as an academic discipline in 1956"

    tokenizer = BertWordPieceTokenizer.from_file(vocab_path)
    tokenizer.enable_padding(length=512)
    tokenizered_text = tokenizer.encode(sentence, add_special_tokens=False)
    logger.info(f"tokenizer results: f{tokenizered_text.ids}")
    
    inference = BaseBiEncoderInference(ckpt_path, config_file, vocab_path)
    embedding = inference.query_single_encode(sentence)
    logger.info(f"query embedding: \n {embedding}")
    
    queries = read_json_lines(query_data_path)
    embeddings = inference.query_batch_encode(queries, 4, use_tokenizer=False)
    np.save(os.path.join(prediction_save_path, "query.npy"), embeddings)


if __name__ == '__main__':
    main()
