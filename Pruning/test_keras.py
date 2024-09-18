import os
import random

import numpy as np
from keras import Model, Sequential, Input
from keras.callbacks import EarlyStopping
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.layers import BatchNormalization, Activation, Conv2D, Dense, Add, UpSampling2D
from keras.saving.save import load_model
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.cluster import MeanShift, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from CGA.CGA import complete_gradient_algorithm
from CGA.cluster_CGA import cluster_algorithm

def prune_filter(original_model, cut_off_layer_name, indexes):
    layer_to_prune = original_model.get_layer(cut_off_layer_name)

    all_layers = original_model.layers

    layer_to_prune_index = all_layers.index(layer_to_prune)

    previous_layer = next((layer for layer in all_layers[:layer_to_prune_index][::-1] if layer.name.endswith("out")),
                          None)
    next_layer = next((layer for layer in all_layers[layer_to_prune_index + 1:] if layer.name.endswith("add")), None)

    model_prev = Model(inputs=original_model.input, outputs=previous_layer.output)
    model_to_prune = Model(inputs=original_model.layers[all_layers.index(previous_layer) + 1].input,
                           outputs=next_layer.output)
    model_next = Model(inputs=original_model.layers[all_layers.index(next_layer) + 1].input,
                       outputs=original_model.output)

    config = layer_to_prune.get_config()
    config['filters'] = len(indexes)


    new_layer = Conv2D(**config)
    input_shape = layer_to_prune.input_shape
    new_layer.build(input_shape)

    weights = layer_to_prune.get_weights()[0][:, :, :, indexes]

    if len(layer_to_prune.get_weights()) > 1:
        biases = layer_to_prune.get_weights()[1][indexes]
        new_weights = [weights, biases]
    else:
        new_weights = [weights]
    new_layer.set_weights(new_weights)

    x = model_prev.output
    shortcut = x

    for layer in model_to_prune.layers:
        if layer.name == cut_off_layer_name:
            x = new_layer(x)
        elif isinstance(layer, Conv2D):
            config = layer.get_config()
            new_layer_conv = Conv2D(**config)
            input_shape = layer.input_shape
            new_layer_conv.build(input_shape)
            if '_3_' in layer.name:
                new_layer_conv = Conv2D(
                    filters=len(indexes),
                    kernel_size=layer.kernel_size,
                    strides=(4,4),
                    padding=layer.padding,
                    activation=None,
                    use_bias=layer.use_bias,
                    kernel_initializer=layer.kernel_initializer,
                    bias_initializer=layer.bias_initializer,
                    name=layer.name
                )
                shortcut = new_layer_conv(shortcut)
            else:
                x = new_layer_conv(x)
        elif isinstance(layer, BatchNormalization):
            new_layer_batch = BatchNormalization(name=layer.name)
            if '_3_' in layer.name:
                if '_3_' in layer.name:
                    if not 'block1' in layer.name:
                        x = new_layer_batch(x)
                    else:
                        shortcut = new_layer_batch(shortcut)
                    x = Add(name=layer.name.replace("_3_bn", "_add"))([x, shortcut])
                else:
                    x = new_layer_batch(x)
            else:
                x = new_layer_batch(x)
        elif isinstance(layer, Activation):
            new_layer_activation = Activation(layer.activation.__name__, name=layer.name)
            x = new_layer_activation(x)
            if 'out' in layer.name:
                shortcut = x

    pruned_model = Model(inputs=model_prev.output, outputs=x)
    for i in range(len(pruned_model.weights)):
        pruned_model.weights[i]._handle_name = pruned_model.weights[i].name + "_" + str(i)

    new_model = Sequential()
    new_model.add(model_prev)
    new_model.add(pruned_model)

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

    new_model.add(model_next)
    return new_model

def extract_feature_maps(model_, layer_name, input_data):
    intermediate_layer_model = Model(inputs=model_.input, outputs=model_.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(input_data)
    return intermediate_output

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
    output_layer = 'conv4_block2_2_conv'
    # model = load_model('../Models/Final_NN/model_stanford_dog_renet50.h5')
    model = load_model('../Models/Final_NN/model_cifar10_renet50.h5')
    # model = load_model('../Models/NN/model_fashion_mnist_renet50_4_categories.h5')

    resnet_model = model.get_layer('resnet50')
    # resnet_model = model
    resnet_model.summary()

    feature_maps = extract_feature_maps(resnet_model, output_layer, np.random.rand(1, 300, 300, 3))

    num_samples, height, width, num_filters = feature_maps.shape
    feature_maps_flattened = feature_maps.reshape((num_samples, height * width, num_filters))

    weights_list = feature_maps_flattened.reshape((num_filters, -1))

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

    # mean_shift = KMeans(34)
    # mean_shift.fit(data)
    # labels = mean_shift.labels_
    #
    # unique_labels = np.unique(labels)

    # indexes = []
    # for label in unique_labels:
    #     indices_in_cluster = np.where(labels == label)[0]
    #     random_index = random.choice(indices_in_cluster)
    #     indexes.append(random_index)
    # indexes = [1,5,8,3]
    # train_generator = training_data_gen.flow_from_directory('../Models/data/images/Images',
    #                                                         target_size=(300, 300),
    #                                                         batch_size=32,
    #                                                         class_mode='categorical')
    # test_generator = test_data_gen.flow_from_directory('../Models/data/test',
    #                                                    target_size=(300, 300),
    #                                                    batch_size=32,
    #                                                    class_mode='categorical')
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    train_y = to_categorical(y_train, 10)
    test_y = to_categorical(y_test, 10)
    indexes = [37, 220, 157, 185, 168, 48, 2, 18, 204, 53, 137, 188, 54, 12, 33, 216, 127, 84, 71, 221, 128, 200, 217, 236, 107, 180, 73, 35, 136, 8, 126]
    print(indexes)

    prune_input = Input(shape=(32, 32, 3))

    pruned_model = prune_filter(resnet_model, output_layer, indexes)

    print_all_layers(pruned_model)


    new_model = Sequential()
    # new_model.add(prune_input)
    new_model.add(pruned_model)
    # new_model.add(model.get_layer('dense'))
    new_model.add(Dense(10, activation='softmax', name='dense_out'))

    print_all_layers(new_model)

    # (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # train_X = np.expand_dims(train_X, axis=-1)
    # train_X = np.repeat(train_X, 3, axis=-1)
    # train_X = np.pad(train_X, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
    # train_y = to_categorical(train_y, 10)

    # (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    #
    # train_X = train_X.reshape((train_X.shape[0], 28, 28))
    # X_test = test_X.reshape((test_X.shape[0], 28, 28))
    # # change the type to float
    # train_X = train_X.astype('float32')
    # X_test = X_test.astype('float32')
    #
    # # convert data to 3 channels
    # train_X = np.stack((train_X,) * 3, axis=-1)
    # test_X = np.stack((X_test,) * 3, axis=-1)
    #
    # train_y = to_categorical(train_y)
    # test_y = to_categorical(test_y)

    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = new_model.fit(x=X_train, y=train_y, epochs=1)
    #
    #
    # test_X = np.expand_dims(test_X, axis=-1)
    # test_X = np.repeat(test_X, 3, axis=-1)
    # test_X = np.pad(test_X, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
    # result = new_model.evaluate(test_X, to_categorical(test_y, 10))
    result = new_model.evaluate(X_test, test_y)
    # history = new_model.fit(train_generator,
    #                     epochs=1,
    #                     validation_data=test_generator)
    #
    # result = new_model.evaluate(test_generator, verbose=2)
    print(result)

    train_loss = history.history['loss'][-1]
    train_acc = history.history['accuracy'][-1]
    test_loss, test_acc = result[0], result[1]
    with open("training_results_cifat10_resnet.txt", "a") as file:
        file.write(f"Model: {'../Models/Final_NN/model_cifar_renet50.h5'}\n")
        file.write(f"Indexes: {indexes}\n")
        file.write(f"Rozmiar z: {len(indexes)}\n")
        file.write(f"Training accuracy: {train_acc}, Training loss: {train_loss}\n")
        file.write(f"Test accuracy: {test_acc}, Test loss: {test_loss}\n")
        file.write("=" * 50 + "\n")

    save_path = '../NN/pruned_model_mnist_renet50_4_categories_kmeans.h5'
    # save_path = '../NN/pruned_model_fashion_mnist_renet50_4_categories_1_layer.h5'
    if os.path.exists(save_path):
        os.remove(save_path)

    new_model.save(save_path, overwrite=True)

if __name__ == "__main__":
    main()