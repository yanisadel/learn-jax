import tensorflow as tf
import tensorflow_datasets as tfds


def load_unprocessed_mnist_data():
    data, info = tfds.load(
        name="mnist", data_dir="/tmp/tfds", as_supervised=True, with_info=True
    )

    data_train = data["train"]
    data_test = data["test"]

    return data_train, data_test, info


def preprocess_data(img, label):
    return (tf.cast(img, tf.float32) / 255.0), label