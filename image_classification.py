import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
print("Imports complete")


def main():
    train_data, train_labels = tf.keras.datasets.fashion_mnist.load_data()[0]
    test_data, test_labels = tf.keras.datasets.fashion_mnist.load_data()[1]
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    train_data_norm = train_data / 255.0
    test_data_norm = test_data / 255.0

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_data_reshaped = train_data_norm.reshape(-1, 28, 28, 1)
    test_data_reshaped = test_data_norm.reshape(-1, 28, 28, 1)

    num_epochs = 10
    model = create_model()
    model, class_weights = compile_model(model)
    model_history = fit_model(
        model, 
        train_data_reshaped,
        train_labels, 
        num_epochs, 
        test_data_reshaped, 
        test_labels,
        class_weights,
        datagen
    )
    
    model.evaluate(test_data_reshaped, test_labels)
    print("Model is now finished training and will make predictions")
    y_probs = model.predict(test_data_reshaped)
    plot_confusion_matrix(test_labels, tf.argmax(y_probs, axis=1), class_names, text_size=7)


def return_mean_absolute_error(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def return_mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def create_model():
    tf.random.set_seed(42)
    model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(10, activation='softmax')
    ])
    return model


def compile_model(model):
    class_weights = {
        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
        5: 1.0, 6: 1.8, 7: 1.0, 8: 1.0, 9: 1.0
    }
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model, class_weights


def fit_model(model, X_train, y_train, num_epochs, X_val, y_val, class_weights, datagen):
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        mode='max'
    )

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=num_epochs,
        validation_data=(X_val, y_val),
        callbacks=[lr_scheduler, early_stopping],
        class_weight=class_weights
    )
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