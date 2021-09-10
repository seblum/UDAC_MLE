from __future__ import print_function

import argparse
import os
import pandas as pd

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# from sklearn.externals import joblib
# Import joblib package directly
import joblib

## Import any additional libraries you need to define a model
from tensorflow import keras
from tensorflow.keras import layers, activations
import tensorflow as tf


# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--criterion', type=str, default='event_offer completed')
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data   = pd.read_csv(os.path.join(training_dir, "train.csv"), names=None)

    crit    = args.criterion
    train_y = train_data.loc[:, [crit]]
    train_x = train_data.drop(crit, axis=1)

    X_train_nn = train_x.values
    y_train_nn = train_y

    epochs = args.epochs
    lr     = args.learning_rate
    
    input_shape, hidden_shape, output_shape = len(X_train_nn[0]), len(X_train_nn[0])*1.5, len(y_train_nn[0])
    
    ## Define a model 
    model = keras.models.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(hidden_shape, activation='relu'),
            layers.Dense(hidden_shape, activation='relu'),
            layers.Dense(output_shape, activation="softmax"),
        ]
    )

    model.summary()
    
    # compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", \
                  metrics=["accuracy"])
    # fit model
    history = model.fit(X_train, y_train, epochs=epochs, verbose=1)

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
   