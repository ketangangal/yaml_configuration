import tensorflow as tf

def get_data(validateion_size):
    dataset = tf.keras.datasets.mnist
    print('here')
    (X_train, y_train), (X_test, y_test) = dataset.load_data()

    X_train_norm = X_train / 255.
    X_test_norm = X_test / 255.

    X_validate_norm, X_test_norm = X_test_norm[0:validateion_size], X_test_norm[validateion_size:]
    y_valid, y_test = y_test[0:validateion_size], y_test[validateion_size:]

    return (X_train_norm, y_train), (X_test_norm, y_test) , (X_validate_norm,y_valid)