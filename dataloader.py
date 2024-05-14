import tensorflow as tf
import tensorflow_datasets as tfds


def load_unprocessed_mnist_data():
    data, info = tfds.load(
        name="mnist", data_dir="/tmp/tfds", as_supervised=True, with_info=True
    )

    data_train = data["train"]
    data_test = data["test"]

    return data_train, data_test, info


def preprocess_sample(img, label):
    return (tf.cast(img, tf.float32) / 255.0), label


def preprocess_data(data, batch_size=32):
    return tfds.as_numpy(data.map(preprocess_sample).batch(batch_size).prefetch(1))


def get_data_info(info):
    height, width, channels = info.features["image"].shape
    num_pixels = height * width * channels
    num_labels = info.features["label"].num_classes
    return height, width, channels, num_pixels, num_labels
