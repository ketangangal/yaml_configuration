from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model
import argparse


def training(config_path):
    config = read_config(config_path)
    validation_size = config["params"]["validation_datasize"]

    (X_train_norm, y_train), (X_test_norm, y_test) , (X_validate_norm,y_valid) = get_data(validation_size)

    input_shape = config["params"]["input_shape"]
    Loss_function = config["params"]["loss_function"]
    optimizer = config["params"]["otimizer"]
    Num_classes = config["params"]["no_classes"]
    metrics = config["params"]["metrics"]

    model = create_model(input_shape, Loss_function, optimizer, metrics, Num_classes)

    Epoch = config["params"]["epochs"]
    history = model.fit(X_train_norm, y_train, validation_data=(X_validate_norm, y_valid), epochs=Epoch)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default=r'C:\Users\ketan\Desktop\DeepLearning\yaml_configuration\config.yaml')
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
