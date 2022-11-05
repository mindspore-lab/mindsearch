# coding=utf-8
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore import ms_class

from mindsearch.bi_encoder.base_model import BertEmbedding, BertConfig
from mindsearch.utils.logger import Logger

logger = Logger(__name__).get_logger()


@dataclass
@ms_class
class EncoderOutput:
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    

class DensePooler(nn.Cell):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True):
        super(DensePooler, self).__init__()
        self.linear_q = nn.Dense(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Dense(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}
        self.output_dim = output_dim

    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, 'pooler.ckpt')
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f'Loading Pooler from {pooler_path}')
                state_dict = ms.load_checkpoint(pooler_path)
                ms.load_param_into_net(self, state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        ms.save_checkpoint(self.state_dict(), os.path.join(save_path, 'pooler.ckpt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)

    def construct(self, q: Tensor = None, p: Tensor = None):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError


class BiEncoderModel(nn.Cell):
    TRANSFORMER_CLS = BertEmbedding

    def __init__(self,
                 lm_q: BertEmbedding,
                 lm_p: BertEmbedding,
                 pooler: nn.Cell = None,
                 untie_encoder: bool = False,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 ):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.untie_encoder = untie_encoder

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            sum_op = ops.ReduceSum()
            s = sum_op(hidden_state * mask.unsqueeze(-1).float(), axis=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        else:
            return None

    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg)
        p_hidden = psg_out[0]
        p_reps = self.sentence_embedding(p_hidden, psg['input_mask'])
        if self.pooler is not None:
            p_reps = self.pooler(p=p_reps)  # D * d
        if self.normlized:
            l2_normalize = ops.L2Normalize(axis=-1)
            p_reps = l2_normalize(p_reps)
        return p_reps

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry)
        q_hidden = qry_out[0]
        q_reps = self.sentence_embedding(q_hidden, qry['input_mask'])
        if self.pooler is not None:
            q_reps = self.pooler(q=q_reps)
        if self.normlized:
            l2_normalize = ops.L2Normalize(axis=-1)
            q_reps = l2_normalize(q_reps)
        return q_reps

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = DensePooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = DensePooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    def construct(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)

        # for inference
        return EncoderOutput(
            q_reps=q_reps,
            p_reps=p_reps,
            loss=None,
            scores=None
        )

    @classmethod
    def load(
            cls,
            model_name_or_path,
            config,
            is_training,
            normlized,
            sentence_pooling_method
    ):
        # load local
        untie_encoder = True
        if os.path.isdir(model_name_or_path):
            # TODO  adapter this branch with no share weights
            _qry_model_path = os.path.join(model_name_or_path, 'query_model.ckpt')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model.ckpt')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(config, is_training, _qry_model_path)
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(config, is_training, _psg_model_path)
                untie_encoder = False
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(config, is_training, model_name_or_path)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(config, is_training, model_name_or_path)
            lm_p = lm_q
            
        pooler_weights = os.path.join(model_name_or_path, 'pooler.ckpt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            untie_encoder=untie_encoder,
            normlized=normlized,
            sentence_pooling_method=sentence_pooling_method,
        )
        return model

