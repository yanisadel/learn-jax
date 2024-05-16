import argparse
import logging
import os

import haiku as hk
import jax
import jax.numpy as jnp
import neptune
import optax
from dotenv import load_dotenv

from learntrix.dataloaders.computer_vision.mnist import load_mnist_dataset
from learntrix.training.losses import cross_entropy_loss
from learntrix.training.supervised_trainers.classification_trainer import (
    ClassificationTrainer,
)
from learntrix.training.supervised_trainers.runners import run_train


def get_parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neptune-project",
        type=str,
        help="Neptune project to which the metrics are logged.",
        default="yanisadel/learn-trix",
    )
    parser.add_argument("--num-steps", type=int, help="Number of steps", default=1000)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=64)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument(
        "--validation-step", type=int, help="Validation step", default=10
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Get command line arguments
    args = get_parser_args()

    # Set default device to CPU for JAX
    jax.config.update("jax_platform_name", "cpu")

    # Get devices if any
    devices = jax.devices("cpu")
    num_devices = len(devices)
    print(f"\nDetected the following devices: {tuple(devices)}")

    # Initializing Neptune
    run = None
    if args.neptune_project != "":
        print("\nInitializing Neptune..")
        load_dotenv("./.env")
        run = neptune.init_run(
            project=args.neptune_project,
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
        )

        params = {"learning_rate": args.lr, "optimizer": "Adam"}
        run["parameters"] = params

    # Load data
    print("\nLoading data..")
    data_train = load_mnist_dataset("train", shuffle=True, batch_size=args.batch_size)
    data_test = load_mnist_dataset("test", shuffle=False, batch_size=10000)

    # Network
    def forward_fn(x: jax.Array) -> jax.Array:
        x = x.astype(jnp.float32) / 255.0
        mlp = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(300),
                jax.nn.relu,
                hk.Linear(100),
                jax.nn.relu,
                hk.Linear(10),
            ]
        )
        return mlp(x)

    # Get trainer
    print("\nInitializing trainer..")
    trainer = ClassificationTrainer(
        forward_fn=forward_fn,
        loss_fn=cross_entropy_loss,
        optimizer=optax.adam(learning_rate=args.lr),
        num_classes=10,
    )

    # Initializing state
    print("\nInitializing state..")
    training_state = trainer.init(
        jax.random.PRNGKey(0), x=jnp.ones(shape=(32, 28, 28, 1))
    )
    training_state = jax.device_put_replicated(training_state, devices)

    # Train
    print("\nTraining..")
    state, metrics = run_train(
        update_fn=trainer.update,
        evaluate_fn=trainer.evaluate,
        state=training_state,
        devices=devices,
        dataset_train=data_train,
        dataset_test=data_test,
        num_steps=args.num_steps,
        validation_step=args.validation_step,
        run_neptune=run,
    )

    print("Training finished")

    # Ending Neptune session
    if run is not None:
        print("\nEnding Neptune session..")
        run.stop()
        print("Done")
