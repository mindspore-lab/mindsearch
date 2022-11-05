# coding=utf-8
import os
import numpy as np

import mindspore
from mindspore import Tensor
from mindsearch.inference.inference import BaseBiEncoderInference
from mindsearch.bi_encoder.faiss_retriever import BaseFaissIPRetriever, search_queries, batch_search_queries
from mindsearch.utils.logger import Logger

mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="CPU")
logger = Logger(__name__).get_logger()


def main():
    model_name_or_path = ""
    config_file = os.path.join(model_name_or_path, "config.json")
    ckpt_path = os.path.join(model_name_or_path, "mindspore.ckpt")
    vocal_path = os.path.join(model_name_or_path, "vocab.txt")
    sentences = "Artificial intelligence was founded as an academic discipline in 1956."

    query_reps_path = ""
    passage_reps_path = ""
    
    inference = BaseBiEncoderInference(ckpt_path, config_file, vocal_path)
    embedding = inference.query_single_encode(sentences)
    logger.info(f"query embedding: \n {embedding}")
    
    p_reps = np.load(os.path.join(passage_reps_path, "passage.npy"))
    p_reps = np.array(p_reps).astype('float32')
    retriever = BaseFaissIPRetriever(p_reps)
    all_scores, psg_indices = search_queries(retriever, Tensor.asnumpy(embedding))
    logger.info(f"query indices: {psg_indices}")
    
    query_reps_path = os.path.join(query_reps_path, "query.npy")
    all_scores, psg_indices = batch_search_queries(retriever, query_reps_path, use_gpu=True)
    logger.info(f"queries indices: {psg_indices}")


if __name__ == '__main__':
    main()
