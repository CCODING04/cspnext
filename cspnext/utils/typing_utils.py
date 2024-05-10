from typing import List, Optional, Union

import torch

from cspnext.structures import InstanceData, PixelData

DeviceType = Union[str, torch.device]

InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]

PixelList = List[PixelData]
OptPixelList = Optional[PixelList]