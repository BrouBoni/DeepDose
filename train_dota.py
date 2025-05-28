import json
import os
import random

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow_addons.optimizers import LAMB

from src.generators import DataGenerator
from src.models import idota
from utils.utils import set_gpu

set_gpu(0)

if __name__ == '__main__':

    ## Define options
    checkpoints_dir = './checkpoints/'
    name = 'unity'
    print(name)
    expr_dir = os.path.join(checkpoints_dir, name)

    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)

    path = "./data/train/"
    path_ckpt = os.path.join(expr_dir, 'weights.ckpt')

    ## Define hyperparameters
    # Training parameters
    batch_size = 4
    num_epochs = 120
    learning_rate = 0.001
    weight_decay = 0.0001

    # Load model and data hyperparameters
    with open('./hyperparam.json', 'r') as hfile:
        param = json.load(hfile)

    # Load data files
    train_split = 0.90
    listIDs = [*range(1581)]

    # Training, validation, test split.
    random.seed(333)
    random.shuffle(listIDs)
    trainIDs = listIDs[:int(round(train_split*len(listIDs)))]
    valIDs = listIDs[int(round(train_split*len(listIDs))):]

    # Calculate or load normalization constants.
    scale = {'y_min':0, 'y_max':0.05,
            'r_min':0, 'r_max':0.05,
            'x_min':-1000, 'x_max':3000}

    with open(os.path.join(expr_dir, 'scale.json'), 'w') as fp:
        json.dump(scale, fp)

    # Initialize generators.
    train_gen = DataGenerator(path, trainIDs, batch_size, scale)
    val_gen = DataGenerator(path, valIDs, batch_size, scale)

    ## Define and train the transformer.
    transformer = idota(
        inshape=param['inshape'],
        steps=param['num_levels'],
        enc_feats=param['enc_feats'],
        num_heads=param['num_heads'],
        num_transformers=param['num_transformers'],
        kernel_size=param['kernel_size']
    )
    transformer.summary()

    # Load weights from checkpoint.
    random.seed()
    # uncomment when retraining
    # transformer.load_weights(path_ckpt)

    # Compile the model.
    optimizer = LAMB(learning_rate=learning_rate, weight_decay_rate=weight_decay)
    transformer.compile(optimizer=optimizer, loss='mse', metrics=[])

    # Callbacks.
    # Save best model at the end of the epoch.
    checkpoint = ModelCheckpoint(
        filepath=path_ckpt,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min')

    tensorboard_callback = TensorBoard(log_dir=os.path.join(expr_dir, 'logs'))

    # Learning rate scheduler. Manually reduce the learning rate.
    sel_epochs = [10,25,40,55,70,85,100,115]
    lr_scheduler = LearningRateScheduler(
        lambda epoch, lr: lr*0.5 if epoch in sel_epochs else lr,
        verbose=2)

    optimizer.learning_rate.assign(learning_rate)
    history = transformer.fit(
        x=train_gen,
        validation_data=val_gen,
        epochs=num_epochs,
        verbose=2,
        callbacks=[checkpoint, lr_scheduler, tensorboard_callback]
        )

    # Save last weights and hyperparameters.
    path_last = os.path.join(expr_dir, 'last.ckpt')
    transformer.save_weights(path_last)
