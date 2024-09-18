import os
import random
import uuid

import cv2
import numpy as np
from keras import Sequential, Model, Input
from keras.callbacks import EarlyStopping
from keras.datasets import fashion_mnist, mnist
from keras.layers import Activation, BatchNormalization, Conv2D, UpSampling2D, Concatenate, SeparableConv2D, Add, \
    GlobalAveragePooling2D, Dropout, Dense
from keras.models import clone_model
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling
from CGA.CGA import complete_gradient_algorithm
from CGA.cluster_CGA import cluster_algorithm

training_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range = 0.12,
    height_shift_range = 0.12,
    width_shift_range = 0.12,
    horizontal_flip = True
)
test_data_gen = ImageDataGenerator(rescale=1./255)

def prune_filter(original_model, cut_off_layer_name, indexes):
    layer_to_prune = original_model.get_layer(cut_off_layer_name)

    all_layers = original_model.layers

    layer_to_prune_index = all_layers.index(layer_to_prune)

    previous_layer = next((layer for layer in all_layers[:layer_to_prune_index][::-1] if layer.name.endswith("_bn")),
                          None)
    next_layer = next((layer for layer in all_layers[layer_to_prune_index + 1:] if layer.name.endswith("_8")), None)

    # previous_layer = next((layer for layer in all_layers[:layer_to_prune_index][::-1] if layer.name.endswith("_add")),
    #                       None)
    # next_layer = next((layer for layer in all_layers[layer_to_prune_index + 1:] if layer.name.endswith("_BN")), None)

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
            # if 'sepconv3' in layer.name:
            #     # new_layer_conv = SeparableConv2D(
            #     #     filters=layer.,
            #     #     kernel_size=layer.kernel_size,
            #     #     padding=layer.padding,
            #     #     activation=None,
            #     #     use_bias=layer.use_bias,
            #     #     kernel_initializer=layer.kernel_initializer,
            #     #     bias_initializer=layer.bias_initializer,
            #     #     name=layer.name,
            #     #     input_shape=layer.input_shape
            #     # )
            #     input_shape = layer.input_shape
            #     new_layer_conv.build(input_shape)
            #     shortcut = new_layer_conv(shortcut)
            # else:
            #     input_shape = new_layer.output_shape
            #     new_layer_conv.build(input_shape)
            x = new_layer_conv(x)
        elif isinstance(layer, BatchNormalization):
            new_layer_batch = BatchNormalization(name=layer.name)
            x = new_layer_batch(x)
        elif isinstance(layer, Activation):
            new_layer_activation = Activation(layer.activation.__name__, name=layer.name)
            x = new_layer_activation(x)
        elif isinstance(layer, Concatenate):
            x = Concatenate()([model_prev.output, x])

    pruned_model = Model(inputs=model_prev.output, outputs=x)
    for i in range(len(pruned_model.weights)):
        pruned_model.weights[i]._handle_name = pruned_model.weights[i].name + "_" + str(i)
    # for i in range(len(pruned_model.weights)):
    #     pruned_model.weights[i]._handle_name = pruned_model.weights[i].name + "_" + str(i)

    new_model = Sequential()
    new_model.add(model_prev)
    new_model.add(rename_layers(pruned_model))

    pruned_output_shape = pruned_model.output_shape[1:]
    model_next_input_shape = model_next.input_shape[1:]

    if pruned_output_shape != model_next_input_shape:
        if pruned_output_shape[-1] != model_next_input_shape[-1]:
            new_model.add(Conv2D(model_next_input_shape[-1], (1, 1), padding='same'))

        if pruned_output_shape[0] != model_next_input_shape[0] or pruned_output_shape[1] != model_next_input_shape[1]:
            new_model.add(UpSampling2D(size=(
                model_next_input_shape[0] // pruned_output_shape[0],
                model_next_input_shape[1] // pruned_output_shape[1]
            )))

            new_model.add(Conv2D(model_next_input_shape[-1], (3, 3), padding='same'))

    new_model.add(rename_layers(model_next))
    return new_model


def extract_feature_maps(model_, layer_name, input_data):
    intermediate_layer_model = Model(inputs=model_.input, outputs=model_.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(input_data)
    return intermediate_output

def print_all_layers(model):
    def print_layer(layer, prefix=''):
        layer_output_shape = layer.output.shape if hasattr(layer, 'output_shape') else 'unknown shape'
        print(f"{prefix}{layer.name} ({layer.__class__.__name__}) - {layer_output_shape}")
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                print_layer(sublayer, prefix=prefix + '  ')

    for layer in model.layers:
        print_layer(layer)

def rename_layers(model, prefix='layer'):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Model):
            # Rekurencyjnie zmieniamy nazwy warstw w zagnieżdżonych modelach
            rename_layers(layer, prefix=f"{prefix}_{i}")
        else:
            new_name = f"{prefix}_{i}_{str(uuid.uuid4())[:8]}"
            layer._name = new_name
    return model


def main():
    # output_layer = 'block_10_expand'
    output_layer = 'conv_pw_7'

    # model = load_model('../Models/Final_NN/model_fashion_mnist_mobilenet.h5')
    model = load_model('../Models/NN/model_mnist_mobilenet.h5')
    mobile_model = model.get_layer('mobilenet_1.00_128')
    print(model.summary())

    # train_generator = training_data_gen.flow_from_directory('../Models/data/images/Images',
    #                                                         target_size=(300, 300),
    #                                                         batch_size=32,
    #                                                         class_mode='categorical')
    # test_generator = test_data_gen.flow_from_directory('../Models/data/test',
    #                                                    target_size=(300, 300),
    #                                                    batch_size=32,
    #                                                    class_mode='categorical')
    (train_x, train_y), (test_X, test_y) = mnist.load_data()

    # Ponieważ MobileNet działa na obrazach o większych rozmiarach, musimy zmienić rozmiar obrazów MNIST z 28x28 na 224x224
    train_x = np.stack([np.stack([img] * 3, axis=-1) for img in train_x])  # Zduplikowanie kanałów do 3
    train_x = np.array([tf.image.resize(img, (128, 128)) for img in train_x])

    test_X = np.stack([np.stack([img] * 3, axis=-1) for img in test_X])  # Zduplikowanie kanałów do 3
    test_X = np.array([tf.image.resize(img, (128, 128)) for img in test_X])

    # Normalizacja danych
    X_train = train_x / 255.0
    X_test = test_X / 255.0

    # Konwersja etykiet do formatu one-hot encoding
    y_train = to_categorical(train_y, 10)
    y_test = to_categorical(test_y, 10)

    # (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    #
    # X_train = np.array([cv2.resize(im, (128, 128)) for im in X_train])
    # X_test = np.array([cv2.resize(im, (128, 128)) for im in X_test])
    #
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    #
    # X_train /= 255.0
    # X_test /= 255.0
    #
    # X_train = np.stack((X_train,) * 3, axis=-1)
    # X_test = np.stack((X_test,) * 3, axis=-1)
    #
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    # feature_maps = extract_feature_maps(model, output_layer, np.random.rand(1, 300, 300, 3))
    # rithm(x, h, s, data)


    indexes = [325, 44, 206, 184, 425, 186, 355]
    # indexes = [215, 314, 184, 190, 59, 275, 96, 193, 39, 345, 75, 24, 18, 313, 343, 140, 149, 22, 372, 186, 180, 373, 367, 185, 10, 100, 11, 80, 20, 240, 324, 141, 377, 353, 77, 366, 23, 381, 91, 221, 81, 244, 19, 362, 6, 364, 382, 61, 0, 363, 383, 365]
    # print(len(indexes))
    # feature_maps = extract_feature_maps(model, output_layer, np.random.rand(1, 300, 300, 3))
    #
    # num_samples, height, width, num_filters = feature_maps.shape
    # feature_maps_flattened = feature_maps.reshape((num_samples, height * width, num_filters))
    #
    # weights_list = feature_maps_flattened.reshape((num_filters, -1))
    # pca = PCA(n_components=3)
    # data = pca.fit_transform(weights_list)
    # print('a')
    #
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    # x, h, s = complete_gradient_algorithm(data)
    #
    # x = scaler.fit_transform(x)
    # z = cluster_algorithm(x, h, s, data)
    #
    # indexes = [random.choice(l) for l in z]
    print(indexes)

    pruned_model = prune_filter(mobile_model, output_layer, indexes)
    print_all_layers(pruned_model)

    # new_model.add(pruned_model)
    # new_model.add(Dense(120, activation='softmax'))
    new_model = pruned_model
    new_model.add(model.get_layer('global_average_pooling2d'))
    new_model.add(model.get_layer('dense'))
    # new_model.add(Dense(120, activation='softmax'))
    new_model.add(Dense(10, activation='softmax', name='dens_fin'))


    # new_model = rename_layers(new_model)
    print_all_layers(new_model)

    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # early_stopping = EarlyStopping(
    #     monitor='accuracy',
    #     patience=1,
    #     restore_best_weights=True
    # )
    # history = new_model.fit(train_generator,
    #                     epochs=1,
    #                     validation_data=test_generator)
    #
    # result = new_model.evaluate(test_generator, verbose=2)
    history = new_model.fit(x=X_train, y=y_train, epochs=1)
    result = new_model.evaluate(X_test, y_test, verbose=2)
    train_loss = history.history['loss'][-1]
    train_acc = history.history['accuracy'][-1]
    test_loss, test_acc = result[0], result[1]
    with open("training_results_mobilenet_stanford_dog.txt", "a") as file:
        file.write(f"Model: {'../Models/Final_NN/mobilenet_sanford_dog.h5'}\n")
        file.write(f"Indexes: {indexes}\n")
        file.write(f"Rozmiar z: {len(indexes)}\n")
        file.write(f"Training accuracy: {train_acc}, Training loss: {train_loss}\n")
        file.write(f"Test accuracy: {test_acc}, Test loss: {test_loss}\n")
        file.write("=" * 50 + "\n")
    # print(f'\nTest accuracy: {test_acc}')
    save_path =  '../NN/pruned_model_fashion_mnist_mobilenet_4_categories_1_layer.h5'
    if os.path.exists(save_path):
        os.remove(save_path)

    new_model.save(save_path, overwrite=True)

if __name__ == "__main__":
    main()