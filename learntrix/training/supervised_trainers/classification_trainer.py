from functools import partial
from typing import Callable, Tuple

import haiku as hk
import jax
import optax

from learntrix.types import Batch, Metrics, RNGKey, TrainingState


class ClassificationTrainer:
    """
    Class used to train a supervised classification model
    """

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
        """
        Args:
            forward_fn (Callable[[hk.Params, jax.Array], jax.Array]):
                Function that performs the forward pass.
            loss_fn (Callable[
                        [
                            hk.Params,
                            Batch,
                            Callable[[hk.Params, jax.Array], jax.Array], int],
                            Tuple[jax.Array, Metrics],
                        ]
                    ]
                ):
                Supervised loss function.
            optimizer (optax.GradientTransformation):
                Optax optimizer.
            num_classes (int, optional):
                The number of classes in the classificaiton problem. Defaults to 2.
        """
        self._forward_fn = hk.without_apply_rng(hk.transform(forward_fn))
        self._loss_fn = partial(
            loss_fn, forward_fn=self._forward_fn.apply, num_classes=num_classes
        )
        self._optimizer = optimizer
        self._num_classes = num_classes

    def init(self, random_key: RNGKey, x: jax.Array) -> TrainingState:
        """
        Initializes the state (parameters of the network and of the optimizer)

        Args:
            random_key (RNGKey): Random Key
            x (jax.Array): Array with the shape of the network's input (including
                           batch size)

        Returns:
            TrainingState: Contains network's parameters and optmizer's parameters
        """
        params = self._forward_fn.init(random_key, x)
        opt_state = self._optimizer.init(params)

        return TrainingState(params, opt_state)

    @partial(jax.jit, static_argnums=0)
    def update(
        self, state: TrainingState, batch: Batch
    ) -> Tuple[TrainingState, Metrics]:
        """
        Performs an update on the TrainingState with one batch.

        Args:
            state (TrainingState): Network's and optimizer's parameters
            batch (Batch): Batch of data (containing samples and associated labels)

        Returns:
            Tuple[TrainingState, Metrics]:
                - TrainingState: The updated TrainingState of the training process.
                - Metrics: A Metrics object with loss and accuracy for the batch
        """
        (_, metrics), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(
            state.params, batch
        )
        grads = jax.lax.pmean(grads, axis_name="batch")
        updates, opt_state = self._optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), metrics

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, state: TrainingState, batch: Batch) -> Metrics:
        """
        Function that evaluates the model on a batch from the validation set.

        Args:
            state (TrainingState): Network's and optimizer's parameters
            batch (Batch): Batch of data (containing samples and associated labels)

        Returns:
            Metrics:  A Metrics object with loss and accuracy for the batch
        """
        _, metrics = self._loss_fn(state.params, batch)

        return metrics
