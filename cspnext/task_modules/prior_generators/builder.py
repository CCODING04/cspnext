from copy import deepcopy

from .point_generator import MlvlPointGenerator


def prior_generator_builder(cfg: dict):
    _cfg = deepcopy(cfg)
    type = _cfg.pop('type', None)
    if type == 'MlvlPointGenerator':
        return MlvlPointGenerator(**_cfg)
    else:
        raise NotImplementedError(f"[Prior Generator] {type} not implemented")