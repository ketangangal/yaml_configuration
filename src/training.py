from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model
from logs import logger


import argparse


def training(config_path):
    config = read_config(config_path)
    path = config["logs"]["logs_dir"]
    log = logger.Logger(file=path)

    log.info(log_type='Info', log_message=' Training Started ')
    validation_size = config["params"]["validation_datasize"]
    log.info(log_type='Info', log_message=f'validation_size :{validation_size}  ')

    (X_train_norm, y_train), (X_test_norm, y_test) , (X_validate_norm,y_valid) = get_data(validation_size,config)
    log.info(log_type='Info', log_message='Train - Test - Validation Split Completed')

    input_shape = config["params"]["input_shape"]
    Loss_function = config["params"]["loss_function"]
    optimizer = config["params"]["otimizer"]
    Num_classes = config["params"]["no_classes"]
    metrics = config["params"]["metrics"]

    log.info(log_type='Info', log_message=f'Model Params: {config["params"]}' )

    model = create_model(input_shape, Loss_function, optimizer, metrics, Num_classes, config)
    log.info(log_type='Info', log_message=f'Model created ')

    Epoch = config["params"]["epochs"]
    history = model.fit(X_train_norm, y_train, validation_data=(X_validate_norm, y_valid), epochs=Epoch)
    log.info(log_type='Info', log_message=f'Model History {history} ')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default=r'C:\Users\ketan\Desktop\DeepLearning\yaml_configuration\config.yaml')
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
