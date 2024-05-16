from typing import Dict, NamedTuple

import haiku as hk
import jax.numpy as jnp
import numpy as np
import optax
from typing_extensions import TypeAlias


class Batch(NamedTuple):
    image: np.ndarray
    label: np.ndarray


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


Metrics: TypeAlias = Dict[str, jnp.ndarray]
