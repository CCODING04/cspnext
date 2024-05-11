from abc import ABCMeta, abstractmethod, abstractproperty, abstractstaticmethod
from typing import List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from torch import BoolTensor, Tensor

T = TypeVar('T')
DeviceType = Union[str, torch.device]
IndexType = Union[slice, int, list, torch.LongTensor, torch.cuda.LongTensor,
                  torch.BoolTensor, torch.cuda.BoolTensor, np.ndarray]


