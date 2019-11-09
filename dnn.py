# Importing core libraries
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib

# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")

# Classifiers
from catboost import CatBoostClassifier, Pool

# Model selection
from sklearn.model_selection import StratifiedKFold

# Metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import make_scorer

# Loading data directly from CatBoost
X = pd.read_csv('train.csv')

Xt = pd.read_csv('test.csv')

y = X["ACTION"].apply(lambda x: 1 if x == 1 else 0).values
X.drop(["ACTION"], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder

label_encoders = [LabelEncoder() for _ in range(X.shape[1])]

for col, column in enumerate(X.columns):
    label_encoders[col].fit(X[column].append(Xt[column]))
    X[column] = label_encoders[col].transform(X[column])
    Xt[column] = label_encoders[col].transform(Xt[column])
def frequency_encoding(column, df, df_test=None):
    frequencies = df[column].value_counts().reset_index()
    df_values = df[[column]].merge(frequencies, how='left',
                                   left_on=column, right_on='index').iloc[:,-1].values
    if df_test is not None:
        df_test_values = df_test[[column]].merge(frequencies, how='left',
                                                 left_on=column, right_on='index').fillna(1).iloc[:,-1].values
    else:
        df_test_values = None
    return df_values, df_test_values

for column in X.columns:
    train_values, test_values = frequency_encoding(column, X, Xt)
    X[column+'_counts'] = train_values
    Xt[column+'_counts'] = test_values
categorical_variables = [col for col in X.columns if '_counts' not in col]
numeric_variables = [col for col in X.columns if '_counts' in col]
print(X.head())

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation, LeakyReLU

# Add the GELU function to Keras
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

get_custom_objects().update({'gelu': Activation(gelu)})

# Add leaky-relu so we can use it as a string
get_custom_objects().update({'leaky-relu': Activation(LeakyReLU(alpha=0.2))})
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.layers import Input, Embedding, Reshape, GlobalAveragePooling1D
from tensorflow.keras.layers import Flatten, concatenate, Concatenate, Lambda, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Reshape, MaxPooling1D,BatchNormalization, AveragePooling1D, Conv1D
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2, l1_l2
from keras.losses import binary_crossentropy

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt


def tabular_dnn(numeric_variables, categorical_variables, categorical_counts,
                feature_selection_dropout=0.2, categorical_dropout=0.1,
                first_dense=256, second_dense=256, dense_dropout=0.2,
                activation_type=gelu):
    numerical_inputs = Input(shape=(len(numeric_variables),))
    numerical_normalization = BatchNormalization()(numerical_inputs)
    numerical_feature_selection = Dropout(feature_selection_dropout)(numerical_normalization)

    categorical_inputs = []
    categorical_embeddings = []
    for category in categorical_variables:
        categorical_inputs.append(Input(shape=[1], name=category))
        category_counts = categorical_counts[category]
        categorical_embeddings.append(
            Embedding(category_counts + 1,
                      int(np.log1p(category_counts) + 1),
                      name=category + "_embed")(categorical_inputs[-1]))

    categorical_logits = Concatenate(name="categorical_conc")([Flatten()(SpatialDropout1D(categorical_dropout)(cat_emb))
                                                               for cat_emb in categorical_embeddings])

    x = concatenate([numerical_feature_selection, categorical_logits])
    x = Dense(first_dense, activation=activation_type)(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(second_dense, activation=activation_type)(x)
    x = Dropout(dense_dropout)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model([numerical_inputs] + categorical_inputs, output)

    return model


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def mAP(y_true, y_pred):
    return tf.py_func(average_precision_score, (y_true, y_pred), tf.double)


def compile_model(model, loss, metrics, optimizer):
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


def plot_keras_history(history, measures):
    """
    history: Keras training history
    measures = list of names of measures
    """
    rows = len(measures) // 2 + len(measures) % 2
    fig, panels = plt.subplots(rows, 2, figsize=(15, 5))
    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    try:
        panels = [item for sublist in panels for item in sublist]
    except:
        pass
    for k, measure in enumerate(measures):
        panel = panels[k]
        panel.set_title(measure + ' history')
        panel.plot(history.epoch, history.history[measure], label="Train " + measure)
        panel.plot(history.epoch, history.history["val_" + measure], label="Validation " + measure)
        panel.set(xlabel='epochs', ylabel=measure)
        panel.legend()

    plt.show(fig)


def to_arrays(x, row_index=None):
    """
    given a pandas dataframe, returns a numpy array for each columns
    """
    if row_index is None:
        return [x.iloc[:, col_index].to_numpy() for col_index in range(x.shape[1])]
    else:
        return [x.iloc[row_index, col_index].to_numpy() for col_index in range(x.shape[1])]


def batch_generator(X, y, numeric, categorical, cv=5, batch_size=64, random_state=None):
    '''
    Returns a batch from X, y
    random_state allows determinism
    different scikit-learn cv startegies are possible
    '''
    folds = len(y) // batch_size
    if isinstance(cv, int):
        kf = StratifiedKFold(n_splits=cv,
                             shuffle=True,
                             random_state=random_state)
    else:
        kf = cv

    while True:
        for _, train_index in kf.split(X, y):
            numeric_input = X[numeric].iloc[train_index].to_numpy(dtype=np.float32)
            categorical_input = to_arrays(X[categorical], train_index)
            target = y[train_index]
            yield [numeric_input] + categorical_input, target


SEED = 42
FOLDS = 5
BATCH_SIZE = 512

measure_to_monitor = 'val_auroc'
modality = 'max'
early_stopping = EarlyStopping(monitor=measure_to_monitor,
                               mode=modality,
                               patience=3,
                               verbose=0)

model_checkpoint = ModelCheckpoint('best.model',
                                   monitor=measure_to_monitor,
                                   mode=modality,
                                   save_best_only=True,
                                   verbose=0)

skf = StratifiedKFold(n_splits=FOLDS,
                      shuffle=True,
                      random_state=SEED)

roc_auc = list()
average_precision = list()
oof = np.zeros(len(X))
best_iteration = list()

categorical_levels = {cat: len(set(X[cat].unique()) | set(Xt[cat].unique())) for cat in categorical_variables}

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    model = tabular_dnn(numeric_variables, categorical_variables,
                        categorical_levels,
                        feature_selection_dropout=0.1,
                        categorical_dropout=0.1,
                        first_dense=256,
                        second_dense=256,
                        dense_dropout=0.1)

    model = compile_model(model, binary_crossentropy, [auroc, mAP], Adam(learning_rate=0.0001))

    train_batch = batch_generator(X.iloc[train_idx],
                                  y[train_idx],
                                  numeric_variables,
                                  categorical_variables,
                                  batch_size=BATCH_SIZE,
                                  random_state=SEED)

    val_batch = batch_generator(X.iloc[test_idx],
                                y[test_idx],
                                numeric_variables,
                                categorical_variables,
                                batch_size=BATCH_SIZE,
                                random_state=SEED)

    train_steps = len(y[train_idx]) // BATCH_SIZE

    validation_steps = len(y[test_idx]) // BATCH_SIZE

    history = model.fit_generator(train_batch,
                                  validation_data=val_batch,
                                  epochs=30,
                                  steps_per_epoch=train_steps,
                                  validation_steps=validation_steps,
                                  callbacks=[model_checkpoint, early_stopping],
                                  class_weight=[1.0, (np.sum(y == 0) / np.sum(y == 1))],
                                  verbose=1)

    print("\nFOLD %i" % fold)

    # plot_keras_history(history, measures = ['auroc', 'loss'])

    best_iteration.append(np.argmax(history.history['val_auroc']) + 1)
    preds = model.predict([X.iloc[test_idx][numeric_variables].to_numpy(dtype=np.float32)]
                          + to_arrays(X.iloc[test_idx][categorical_variables]),
                          verbose=1,
                          batch_size=1024).flatten()

    oof[test_idx] = preds

    roc_auc.append(roc_auc_score(y_true=y[test_idx], y_score=preds))
    average_precision.append(average_precision_score(y_true=y[test_idx], y_score=preds))

print("Average cv roc auc score %0.3f ± %0.3f" % (np.mean(roc_auc), np.std(roc_auc)))
print("Average cv roc average precision %0.3f ± %0.3f" % (np.mean(average_precision), np.std(average_precision)))

print("Roc auc score OOF %0.3f" % roc_auc_score(y_true=y, y_score=oof))
print("Average precision OOF %0.3f" % average_precision_score(y_true=y, y_score=oof))


train_batch = batch_generator(X, y,
                              numeric_variables,
                              categorical_variables,
                              batch_size=BATCH_SIZE,
                              random_state=SEED)

train_steps = len(y) // BATCH_SIZE

history = model.fit_generator(train_batch,
                              epochs=int(np.median(best_iteration)),
                              steps_per_epoch=train_steps,
                              class_weight=[1.0, (np.sum(y==0) / np.sum(y==1))],
                              verbose=1)



preds = model.predict([Xt[numeric_variables].to_numpy(dtype=np.float32)]
                      + to_arrays(Xt[categorical_variables]),
                      verbose=1,
                      batch_size=1024).flatten()

submission = pd.DataFrame(Xt.id)
submission['Action'] = preds
submission.to_csv("tabular_dnn_submission.csv", index=False)

tabular_dnn_submission = submission.copy()
