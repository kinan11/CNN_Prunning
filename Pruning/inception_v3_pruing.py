import os
import re
import random
import uuid

import cv2
import numpy as np
from keras import Sequential, Model, Input
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10, mnist
from keras.models import clone_model
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, UpSampling2D, Dense, SeparableConv2D, Add, \
    GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from CGA.CGA import complete_gradient_algorithm
from CGA.cluster_CGA import cluster_algorithm


def prune_filter(original_model, cut_off_layer_name, indexes):
    layer_to_prune = original_model.get_layer(cut_off_layer_name)

    all_layers = original_model.layers

    layer_to_prune_index = all_layers.index(layer_to_prune)

    previous_layer = next((layer for layer in all_layers[:layer_to_prune_index][::-1] if layer.name.endswith("mixed8")),
                          None)
    next_layer = next((layer for layer in all_layers[layer_to_prune_index + 1:] if layer.name.endswith("activation_80")), None)

    # previous_layer = next((layer for layer in all_layers[:layer_to_prune_index][::-1] if layer.name.endswith("mixed8")),
    #                       None)
    # next_layer = next(
    #     (layer for layer in all_layers[layer_to_prune_index + 1:] if layer.name.endswith("activation_268")), None)

    model_prev = Model(inputs=original_model.input, outputs=previous_layer.output)
    model_to_prune = Model(inputs=original_model.layers[all_layers.index(previous_layer) + 1].input,
                           outputs=next_layer.output)
    model_next = Model(inputs=original_model.layers[all_layers.index(next_layer) + 1].input,
                       outputs=original_model.layers[-2].output)

    config = layer_to_prune.get_config()
    config['filters'] = len(indexes)


    new_layer = Conv2D(**config)
    input_shape = layer_to_prune.input.shape
    new_layer.build(input_shape)

    weights = layer_to_prune.get_weights()[0][:, :, :, indexes]

    if len(layer_to_prune.get_weights()) > 1:
        biases = layer_to_prune.get_weights()[1][indexes]
        new_weights = [weights, biases]
    else:
        new_weights = [weights]
    new_layer.set_weights(new_weights)

    x = model_prev.output

    for layer in model_to_prune.layers:
        if layer.name == cut_off_layer_name:
            x = new_layer(x)
        elif isinstance(layer, Conv2D):
            config = layer.get_config()
            new_layer_conv = Conv2D(**config)
            input_shape = layer.input.shape
            new_layer_conv.build(input_shape)
            x = new_layer_conv(x)
        elif isinstance(layer, BatchNormalization):
            new_layer_batch = BatchNormalization(name=layer.name)
            x = new_layer_batch(x)
        elif isinstance(layer, Activation):
            new_layer_activation = Activation(layer.activation.__name__, name=layer.name)
            x = new_layer_activation(x)

    pruned_model = Model(inputs=model_prev.output, outputs=x)
    for i in range(len(pruned_model.weights)):
        pruned_model.weights[i]._handle_name = pruned_model.weights[i].name + "_" + str(i)

    new_model = Sequential()
    new_model.add(model_prev)
    new_model.add(pruned_model)

    pruned_output_shape = pruned_model.output.shape[1:]
    model_next_input_shape = model_next.input.shape[1:]

    if pruned_output_shape != model_next_input_shape:
        if pruned_output_shape[-1] != model_next_input_shape[-1]:
            new_model.add(Conv2D(model_next_input_shape[-1], (1, 1), padding='same'))

        if pruned_output_shape[0] != model_next_input_shape[0] or pruned_output_shape[1] != model_next_input_shape[1]:
            new_model.add(UpSampling2D(size=(
                model_next_input_shape[0] // pruned_output_shape[0],
                model_next_input_shape[1] // pruned_output_shape[1]
            )))

            new_model.add(Conv2D(model_next_input_shape[-1], (3, 3), padding='same'))

    new_model.add(model_next)
    return new_model


def extract_feature_maps(model_, layer_name, input_data):
    intermediate_layer_model = Model(inputs=model_.input, outputs=model_.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(input_data)
    return intermediate_output

# def rename_layers(model, prefix='layer'):
#     for layer in model.layers:
#         if isinstance(layer, Model):
#             rename_layers(layer, prefix)
#         else:
#             new_name = f"{prefix}_{str(uuid.uuid4())[:8]}"
#             layer._name = new_name

def rename_layers(model, prefix='layer'):
    new_model = clone_model(model)
    for i, layer in enumerate(new_model.layers):
        if isinstance(layer, Model):
            # Rekurencyjnie zmieniamy nazwy warstw w zagnieżdżonych modelach
            layer = rename_layers(layer, prefix=f"{prefix}_{i}")
        else:
            new_name = f"{prefix}_{i}_{str(uuid.uuid4())[:8]}"
            layer._name = new_name
    return new_model

def print_all_layers(model):
    def print_layer(layer, prefix=''):
        layer_output_shape = layer.output_shape if hasattr(layer, 'output_shape') else 'unknown shape'
        print(f"{prefix}{layer.name} ({layer.__class__.__name__}) - {layer_output_shape}")
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                print_layer(sublayer, prefix=prefix + '  ')

    for layer in model.layers:
        print_layer(layer)

training_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range = 0.12,
    height_shift_range = 0.12,
    width_shift_range = 0.12,
    horizontal_flip = True
)
test_data_gen = ImageDataGenerator(rescale=1./255)

def main():
    output_layer = 'conv2d_80'
    # output_layer = 'conv2d_268'
    # model = load_model('../Models/NN/inception_cifar10.h5')
    model = load_model('../Models/Final_NN/inception_cifar10.h5')
    inception_model = model.get_layer('inception_v3')
    # inception_model = model
    print(inception_model.summary())

    # train_generator = training_data_gen.flow_from_directory('../Models/data/images/Images',
    #                                                         target_size=(300, 300),
    #                                                         batch_size=32,
    #                                                         class_mode='categorical')
    # test_generator = test_data_gen.flow_from_directory('../Models/data/test',
    #                                                    target_size=(300, 300),
    #                                                    batch_size=32,
    #                                                    class_mode='categorical')

    # feature_maps = extract_feature_maps(xception, output_layer, np.random.rand(1, 224, 224, 3))

    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #
    # # Resize MNIST images to 32x32 and add RGB channels
    # X_train = np.array([cv2.resize(img, (75, 75)) for img in X_train])
    # X_test = np.array([cv2.resize(img, (75, 75)) for img in X_test])
    #
    # # Convert grayscale to RGB (by repeating the single channel 3 times)
    # X_train = np.stack([X_train] * 3, axis=-1)  # Add 3 channels (RGB)
    # X_test = np.stack([X_test] * 3, axis=-1)
    #
    # # Normalize the input data
    # X_train = X_train / 255.0
    # X_test = X_test / 255.0
    #
    # # One-hot encode the labels
    # y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # feature_maps = extract_feature_maps(inception_model, output_layer, np.random.rand(1, 75, 75, 3))
    # # feature_maps = extract_feature_maps(inception_model, output_layer, np.random.rand(1, 28, 28, 3))
    #
    # num_samples, height, width, num_filters = feature_maps.shape
    # feature_maps_flattened = feature_maps.reshape((num_samples, height * width, num_filters))
    #
    # weights_list = feature_maps_flattened.reshape((num_filters, -1))
    # pca = PCA(n_components=3)
    # data = pca.fit_transform(weights_list)
    #
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    # x, h, s = complete_gradient_algorithm(data)
    #
    # x = scaler.fit_transform(x)
    # z = cluster_algorithm(x, h, s, data)
    #
    # indexes = [random.choice(l) for l in z]
    indexes = [230, 204, 52, 51, 57, 400, 1, 36, 8, 43, 442, 393]
    # print(indexes)
    # mean_shift = KMeans(9)
    # mean_shift.fit(data)
    # labels = mean_shift.labels_
    #
    # unique_labels = np.unique(labels)
    #
    # indexes = []
    # for label in unique_labels:
    #     indices_in_cluster = np.where(labels == label)[0]
    #     random_index = random.choice(indices_in_cluster)
    #     indexes.append(random_index)
    # indexes = [1,5,8,3]
    print(indexes)

    pruned_model = prune_filter(inception_model, output_layer, indexes)
    #
    # print_all_layers(pruned_model)
    #

    # prune_input = Input(shape=(299, 299, 3))
    new_model = Sequential()
    m = Model(inputs=model.input, outputs=model.get_layer('tf.math.subtract').output) # cifar
    new_model.add(m) # cifar
    # new_model.add(prune_input)
    new_model.add(pruned_model)
    # new_model.add(Dense(120, activation='softmax'))
    # new_model = pruned_model

    new_model.add(model.get_layer('global_average_pooling2d'))
    new_model.add(model.get_layer('dropout'))
    new_model.add(Dense(10, activation='softmax'))
    # new_model.add(Dense(120, activation='softmax'))
    # new_model = rename_layers(new_model)

    # print_all_layers(new_model)
    print(new_model.summary())


    # new_model = rename_layers(new_model)
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # early_stopping = EarlyStopping(
    #     monitor='accuracy',
    #     patience=1,
    #     restore_best_weights=True
    # )
    # new_model.fit(train_generator,
    #               epochs=1,
    #               validation_data=test_generator)

    # test_loss, test_acc = new_model.evaluate(test_generator, verbose=2)
    history = new_model.fit(x=X_train, y=y_train, epochs=1)
    result = new_model.evaluate(X_test, y_test, verbose=2)
    train_loss = history.history['loss'][-1]
    train_acc = history.history['accuracy'][-1]
    test_loss, test_acc = result[0], result[1]
    with open("training_results_inception_mnist.txt", "a") as file:
        file.write(f"Model: {'../Models/Final_NN/inception_mnist.h5'}\n")
        file.write(f"Indexes: {indexes}\n")
        file.write(f"Rozmiar z: {len(indexes)}\n")
        file.write(f"Training accuracy: {train_acc}, Training loss: {train_loss}\n")
        file.write(f"Test accuracy: {test_acc}, Test loss: {test_loss}\n")
        file.write("=" * 50 + "\n")
    save_path = '../NN/pruned_mnist_inceptionV3.h5'
    if os.path.exists(save_path):
        os.remove(save_path)

    new_model.save(save_path, overwrite=True)

if __name__ == "__main__":
    main()