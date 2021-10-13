import tensorflow as tf
from logs import logger


def get_data(validation_size,config):
    path = config["logs"]["logs_dir"]
    log = logger.Logger(file=path)

    dataset = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = dataset.load_data()
    log.info(log_type="Info", log_message="Data fetched from Keras repo")

    X_train_norm = X_train / 255.
    X_test_norm = X_test / 255.
    log.info(log_type="Info", log_message=" Normalization Completed ")

    X_validate_norm, X_test_norm = X_test_norm[0:validation_size], X_test_norm[validation_size:]
    y_valid, y_test = y_test[0:validation_size], y_test[validation_size:]

    return (X_train_norm, y_train), (X_test_norm, y_test) , (X_validate_norm,y_valid)