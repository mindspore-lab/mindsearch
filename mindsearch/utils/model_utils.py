# coding=utf-8
import math

from mindspore import Tensor, ops
from mindspore.train.callback import Callback
from mindsearch.utils.logger import Logger

logger = Logger(__name__).get_logger()


def cosine_similarity(a: Tensor, b: Tensor):
    if not isinstance(a, Tensor):
        a = Tensor(a)
    
    if not isinstance(b, Tensor):
        b = Tensor(b)
    
    if len(a.shape) == 1:
        a = a.expand_dims(axis=0)
    
    if len(b.shape) == 1:
        b = b.expand_dims(axis=0)
    
    l2_normalize = ops.L2Normalize(axis=1)
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    
    matmul = ops.MatMul(transpose_b=True)
    return matmul(a_norm, b_norm)


def generate_total_layers_params(total_layers,
                                 mindspore_params_per_layer,
                                 torch_params_per_layer,
                                 mindspore_additional_params,
                                 torch_additional_params):
    """
    Generate the total parameter mapping of mindspore and pytorch.

    Args:
        total_layers(int): The total layers of the net.
        mindspore_params_per_layer(list): The list of params per layer for the net of mindspore.
        torch_params_per_layer(list): The list of params per layer for the net of pytorch.
        mindspore_additional_params(list): The list of params outside the layer for the net of mindspore
        torch_additional_params(list): The list  of params outside the layer for the net of pytorch.

    Returns:
        A list of tuple. The first element is the parameter name of mindspore,
        the another is the parameter name of pytorch.
    """
    mapped_params = list(zip(mindspore_params_per_layer, torch_params_per_layer))
    ms_extend_param_list = []
    torch_extend_param_list = []
    for i in range(total_layers):
        for ms_para, torch_para in mapped_params:
            src = ms_para.format(i)
            tgt = torch_para.format(i)

            ms_extend_param_list.append(src)
            torch_extend_param_list.append(tgt)

    mapped_params = list(zip(mindspore_additional_params, torch_additional_params))
    for ms_para, torch_para in mapped_params:
        ms_extend_param_list.append(ms_para)
        torch_extend_param_list.append(torch_para)

    return list(zip(ms_extend_param_list, torch_extend_param_list))


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, dataset_size=-1):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size

    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            log_str = "epoch: {}, current epoch percent: {}, step: {}, outputs are {}".\
                          format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, str(cb_params.net_outputs))
            logger.info(log_str)
        else:
            log_str = "epoch: {}, step: {}, outputs are {}".\
                format(cb_params.cur_epoch_num, cb_params.cur_step_num, str(cb_params.net_outputs))
            logger.info(log_str)
