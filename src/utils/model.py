import tensorflow as tf
from logs import logger
import os
import time


def create_model(input_shape, Loss_function, optimizer, metrics, Num_classes, config):
    path = config["logs"]["logs_dir"]
    log = logger.Logger(file=path)

    LAYERS = [
        tf.keras.layers.Flatten(input_shape=input_shape, name='Input_Layer'),
        tf.keras.layers.Dense(500, activation='relu', name='1st_Hidden_Layer'),
        tf.keras.layers.Dense(300, activation='relu', name='2nd_Hidden_Layer'),
        tf.keras.layers.Dense(100, activation='relu', name='3nd_Hidden_Layer'),
        tf.keras.layers.Dense(Num_classes, activation='softmax', name='Output_Layer')
    ]
    log.info(log_type='Info', log_message=' Layers Initialization done ')
    model = tf.keras.models.Sequential(LAYERS)

    log.info(log_type='Info', log_message=f' model Summary \n {model.summary()}')

    model.compile(loss=Loss_function,
                  optimizer=optimizer,
                  metrics=metrics)

    # untrained model
    return model


def get_unique_filename(filename):
    unique_filename = time.strftime(f"{filename}._%Y_%m_%d_%H_%M_%S.h5")
    return unique_filename


def save_model(model, name, path):
    unique_name = get_unique_filename(name)
    path_to_model = os.path.join(path, unique_name)
    model.save(path_to_model)