from typing import Callable, Iterator, List, Optional, Tuple

import jax
import jaxlib
import neptune
from tqdm import tqdm

from learntrix.types import Batch, Metrics, TrainingState


def run_train(
    update_fn: Callable[[TrainingState, Batch], Tuple[TrainingState, Metrics]],
    evaluate_fn: Callable[[TrainingState, Batch], Metrics],
    state: TrainingState,
    devices: List[jaxlib.xla_extension.Device],
    dataset_train: Iterator[Batch],
    dataset_test: Iterator[Batch],
    num_steps: int,
    validation_step: int,
    run_neptune: Optional[neptune.metadata_containers.run.Run] = None,
) -> Tuple[TrainingState, Metrics]:

    all_metrics: Metrics = {
        "train_loss": [],
        "train_acc": [],
        "train_step": [],
        "val_loss": [],
        "val_acc": [],
        "val_step": [],
    }

    batch_test = next(dataset_test)
    batch_test = jax.device_put_replicated(batch_test, devices)

    for step in tqdm(range(1, num_steps + 1)):
        batch = next(dataset_train)
        batch = jax.device_put_replicated(batch, devices)
        """
        batch = jax.tree.map(
            lambda x: x.reshape((len(devices), -1) + x.shape[1:]),
            batch
            )
        """
        state, metrics = jax.pmap(update_fn, devices=devices, axis_name="batch")(
            state, batch
        )

        train_loss = jax.device_get(metrics["loss"]).mean()
        train_accuracy = jax.device_get(metrics["accuracy"]).mean()

        if run_neptune is not None:
            run_neptune["train/loss"].log(train_loss, step=step)
            run_neptune["train/accuracy"].log(train_accuracy, step=step)

        all_metrics["train_loss"].append(train_loss)
        all_metrics["train_acc"].append(train_accuracy)
        all_metrics["train_step"].append(step)

        if step % validation_step == 0:
            """
            batch_test = jax.tree.map(
                lambda x: x.reshape((len(devices), -1) + x.shape[1:]),
                batch_test
                )
            """
            metrics = jax.pmap(evaluate_fn, devices=devices, axis_name="batch")(
                state, batch_test
            )
            test_loss = jax.device_get(metrics["loss"]).mean()
            test_accuracy = jax.device_get(metrics["accuracy"]).mean()

            if run_neptune is not None:
                run_neptune["test/loss"].log(test_loss, step=step)
                run_neptune["test/accuracy"].log(test_accuracy, step=step)

            all_metrics["val_loss"].append(test_loss)
            all_metrics["val_acc"].append(test_accuracy)
            all_metrics["val_step"].append(step)

    return state, all_metrics
