import tensorflow as tf

def create_model(input_shape,Loss_function,optimizer,metrics,Num_classes):
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=input_shape, name='Input_Layer'),
        tf.keras.layers.Dense(500, activation='relu', name='1st_Hidden_Layer'),
        tf.keras.layers.Dense(300, activation='relu', name='2nd_Hidden_Layer'),
        tf.keras.layers.Dense(100, activation='relu', name='3nd_Hidden_Layer'),
        tf.keras.layers.Dense(Num_classes, activation='softmax', name='Output_Layer')
    ]

    model = tf.keras.models.Sequential(LAYERS)

    model.summary()

    model.compile(loss=Loss_function,
                  optimizer=optimizer,
                  metrics=metrics)

    # untrained model
    return model