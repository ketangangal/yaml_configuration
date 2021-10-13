from src.utils.common import read_config
from src.utils.data_mgmt import get_data
import argparse

def training(config_path):
    config = read_config(config_path)
    validateion_size = config["params"]["validation_datasize"]
    (X_train_norm, y_train), (X_test_norm, y_test) , (X_validate_norm,y_valid) = get_data(validateion_size)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default=r'C:\Users\ketan\Desktop\DeepLearning\yaml_configuration\config.yaml')
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
