import tensorflow as tf
from tensorflow import keras

    model = keras.Sequential()
    # First hidden layer.
    model.add(
        keras.layers.Dense(
            hparams[HP_NUM_UNITS_1], activation="relu", input_shape=[X_train.shape[1]]
        )
    )
    model.add(keras.layers.Dropout(hparams[HP_DROPOUT_1]))
    model.add(keras.layers.Dense(hparams[HP_NUM_UNITS_2], activation="relu"))
    model.add(keras.layers.Dropout(hparams[HP_DROPOUT_2]))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # Display model summary
    model.summary()

    # Initialize optimizer with learning rate.
    if hparams[HP_OPTIMIZER] == "adam":
        optim = keras.optimizers.Adam(learning_rate=hparams[HP_OPTIM_LR])
    elif hparams[HP_OPTIMIZER] == "sgd":
        optim = keras.optimizers.SGD(learning_rate=hparams[HP_OPTIM_LR])

    # Compile the model.
    model.compile(
        optimizer=optim,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.TruePositives(),
        ],
    )

    # Callbacks
    # Early Stopping
    #   -monitor validation loss.
    #   -when validation loss stops decreasing, stop.
    #   -patience is number of epochs with no improvement.
    cb_es = keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=20, verbose=2
    )
    # Model Checkpoint
    #   -call our model "best_model.h5".
    #   -monitor validation loss.
    #   -when validation loss stops decreasing, stop.
    #   -save the best overall model.
    cb_ckpt = keras.callbacks.ModelCheckpoint(
        "best_model_small_2lyr.h5",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=2,
    )

    # Fit
    model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        callbacks=[cb_es, cb_ckpt],
        epochs=200,
        verbose=2,
    )

    _, test_acc, test_fn, test_fp, test_tn, test_tp = model.evaluate(
        X_test, y_test, verbose=2
    )

    return test_acc, test_fn, test_fp, test_tn, test_tp


# For each run, log an hparams summary with hyperparameters and metrics.
def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial.
        accuracy, fn, fp, tn, tp = validate_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
        tf.summary.scalar(METRIC_FN, fn, step=1)
        tf.summary.scalar(METRIC_FP, fp, step=1)
        tf.summary.scalar(METRIC_TN, tn, step=1)
        tf.summary.scalar(METRIC_TP, tp, step=1)
        tf.summary.scalar(METRIC_F1, f1_score(np.array([tn, fp, fn, tp])), step=1)
        tf.summary.scalar(METRIC_F2, f2_score(np.array([tn, fp, fn, tp])), step=1)


# Grid search over parameters and log values.
session_num = 0

for num_units_1 in HP_NUM_UNITS_1.domain.values:
    for dropout_1 in np.arange(
        HP_DROPOUT_1.domain.min_value, HP_DROPOUT_1.domain.max_value + 0.1, 0.2
    ):
        for num_layers in HP_NUM_LAYERS.domain.values:
            for num_units_2 in HP_NUM_UNITS_2.domain.values:
                for dropout_2 in np.arange(
                    HP_DROPOUT_2.domain.min_value,
                    HP_DROPOUT_2.domain.max_value + 0.1,
                    0.2,
                ):
                    for optimizer in HP_OPTIMIZER.domain.values:
                        for optim_lr in HP_OPTIM_LR.domain.values:
                            hparams = {
                                HP_NUM_UNITS_1: num_units_1,
                                HP_DROPOUT_1: dropout_1,
                                HP_NUM_LAYERS: num_layers,
                                HP_NUM_UNITS_2: num_units_2,
                                HP_DROPOUT_2: dropout_2,
                                HP_OPTIMIZER: optimizer,
                                HP_OPTIM_LR: optim_lr,
                            }
                            run_name = "run-%d" % session_num
                            print(f"--- Starting trial: {run_name}")
                            print({h.name: hparams[h] for h in hparams})
                            run(log_dir + run_name, hparams)
                            session_num += 1