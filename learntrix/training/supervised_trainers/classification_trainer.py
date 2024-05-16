from functools import partial
from typing import Callable, Tuple

import haiku as hk
import jax
import optax

from learntrix.types import Batch, Metrics, RNGKey, TrainingState


class ClassificationTrainer:
    def __init__(
        self,
        forward_fn: Callable[[hk.Params, jax.Array], jax.Array],
        loss_fn: Callable[
            [hk.Params, Batch, Callable[[hk.Params, jax.Array], jax.Array], int],
            Tuple[jax.Array, Metrics],
        ],
        optimizer: optax.GradientTransformation,
        num_classes: int = 2,
    ):
        self._forward_fn = hk.without_apply_rng(hk.transform(forward_fn))
        self._loss_fn = partial(
            loss_fn, forward_fn=self._forward_fn.apply, num_classes=num_classes
        )
        self._optimizer = optimizer
        self._num_classes = num_classes

    def init(self, random_key: RNGKey, x: jax.Array) -> TrainingState:
        params = self._forward_fn.init(random_key, x)
        opt_state = self._optimizer.init(params)

        return TrainingState(params, opt_state)

    @partial(jax.jit, static_argnums=0)
    def update(
        self, state: TrainingState, batch: Batch
    ) -> Tuple[TrainingState, Metrics]:
        (_, metrics), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(
            state.params, batch
        )
        grads = jax.lax.pmean(grads, axis_name="batch")
        updates, opt_state = self._optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), metrics

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, state: TrainingState, batch: Batch) -> Metrics:
        _, metrics = self._loss_fn(state.params, batch)

        return metrics
