import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.datasets import make_circles
from sklearn.metrics import confusion_matrix
import itertools
import random

def main():
    train_data, train_labels = tf.keras.datasets.fashion_mnist.load_data()[0]
    test_data, test_labels = tf.keras.datasets.fashion_mnist.load_data()[1]
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    train_data_norm = train_data / 255.0
    test_data_norm = test_data / 255.0

    num_epochs = 10
    model = create_model()
    model = compile_model(model)
    model_history = fit_model(model, train_data_norm, train_labels, num_epochs, test_data_norm, test_labels)
    model.evaluate(test_data_norm, test_labels)

    y_probs = model.predict(test_data_norm)
    plot_confusion_matrix(test_labels, tf.argmax(y_probs, axis=1), class_names, text_size=7)

def return_mean_absolute_error(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def return_mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def create_model():
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
    ])
    return model

def compile_model(model):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"])
    return model

def fit_model(model, X_test, y_test, num_epochs, X_eval, y_eval):
    history = model.fit(X_test, y_test, epochs=num_epochs, validation_data=(X_eval, y_eval))
    return history

def plot_confusion_matrix(y_test, y_pred, classes, text_size=10):
    figsize = (10,10)

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:,np.newaxis]
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    ax.set(title="Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels)

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ax.yaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.title.set_size(20)

    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
            horizontalalignment="center",
            color="white" if cm[i,j] > threshold else "black",
            size=text_size)

    plt.show()

def plot_predictions(**kwargs):
    X_train = kwargs.get("X_training")
    y_train = kwargs.get("y_training")
    X_test = kwargs.get("X_testing")
    y_test = kwargs.get("y_testing")
    preds = kwargs.get("predictions")
    plt.figure(figsize=(10,7))
    plt.scatter(X_train, y_train, c="b", label="Training data")
    plt.scatter(X_test, y_test, c="g", label="Testing data")
    plt.scatter(X_test, preds, c="r", label="Predictions")
    plt.legend();

if __name__ == "__main__":
    main()