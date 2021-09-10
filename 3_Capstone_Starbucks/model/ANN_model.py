from __future__ import print_function

import argparse
import os
import pandas as pd


## Import any additional libraries you need to define a model
from tensorflow import keras
from tensorflow.keras import layers, activations
import tensorflow as tf


if __name__ == '__main__':
        
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--criterion', type=str, default='event_offer completed')
    
    # args holds all passed-in arguments
    args, _ = parser.parse_known_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data   = pd.read_csv(os.path.join(training_dir, "train.csv"), names=None)

    crit    = args.criterion
    train_y = train_data.loc[:, [crit]]
    train_x = train_data.drop(crit, axis=1)

    X_train_nn = train_x.values
    y_train_nn = train_y.values.ravel()


    epochs = args.epochs
    lr     = args.learning_rate
    model_dir  = args.model_dir

    input_shape, hidden_shape, output_shape = len(X_train_nn[0]), 42, 1
    
    ## Define a model 
    model = keras.models.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(hidden_shape, activation='sigmoid'),
            layers.Dense(hidden_shape, activation='sigmoid'),
            layers.Dense(hidden_shape, activation='sigmoid'),
            layers.Dense(hidden_shape, activation='sigmoid'),
            layers.Dense(output_shape, activation="sigmoid"),
            layers.Dropout(0.25)
        ]
    )

    model.summary()
    
    # compile model
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=["accuracy"])
    # fit model
    model.fit(X_train_nn, y_train_nn, epochs=epochs,verbose=1)
   
    tf.contrib.saved_model.save_keras_model(model, '/opt/ml/model')
