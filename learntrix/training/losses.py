from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from learntrix.types import Batch, Metrics


def cross_entropy_loss(
    params: hk.Params,
    batch: Batch,
    forward_fn: Callable[[hk.Params, jax.Array], jax.Array],
    num_classes: int,
) -> Tuple[jax.Array, Metrics]:
    """
    Cross entropy loss
    """
    logits = forward_fn(params, batch.image)
    labels = jax.nn.one_hot(batch.label, num_classes)
    loss_value = -jnp.sum(labels * jax.nn.log_softmax(logits)) / batch.image.shape[0]

    predictions_proba = jax.nn.softmax(logits, axis=-1)
    predictions = jnp.argmax(predictions_proba, axis=-1)
    accuracy = jnp.mean(predictions == batch.label)
    pmean_accuracy = jax.lax.pmean(
        accuracy, axis_name="batch"
    )  # Faudra que je vérifie quand num_devices > 1

    metrics: Metrics = {
        "predictions": predictions_proba,
        "loss": loss_value,
        "accuracy": pmean_accuracy,
    }

    return loss_value, metrics
