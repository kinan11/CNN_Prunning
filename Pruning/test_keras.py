import random

import numpy as np
from keras import Model, Sequential, Input
from keras.datasets import mnist
from keras.layers import BatchNormalization, Activation, Conv2D, Dense
from keras.saving.save import load_model
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sympy import Add

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

    # for layer in model_prev.layers:
    #     layer.trainable = False
    #
    # for layer in model_next.layers:
    #     layer.trainable = False

    combined_input = Input(shape=(32, 32, 3))

    output_prev = model_prev(combined_input)

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

    # input_tensor = Input(shape=model_prev.output_shape)
    # output_prev = model_prev(input_tensor)

    x = model_prev.output

    for layer in model_to_prune.layers:
        if layer.name == cut_off_layer_name:
            x = new_layer(x)
            # x = Conv2D(filters=256, kernel_size=(1, 1), name='adaptation_conv')(x)
        elif isinstance(layer, Conv2D):
            new_layer_conv = Conv2D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                activation=None,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer
            )
            # new_layer.trainable = False
            x = new_layer_conv(x)
        elif isinstance(layer, BatchNormalization):
            new_layer_batch = BatchNormalization()
            # new_layer.trainable = False
            x = new_layer_batch(x)
        elif isinstance(layer, Activation):
            new_layer_activation = Activation(layer.activation)
            # new_layer.trainable = False
            x = new_layer_activation(x)
        elif isinstance(layer, Add):
            index_conv4_block1_add = model_to_prune.layers.index(layer)
            index_conv4_block1_0_bn = index_conv4_block1_add - 2
            index_conv4_block1_3_bn = index_conv4_block1_add - 1
            output_conv4_block1_0_bn = model_to_prune.layers[index_conv4_block1_0_bn].output
            output_conv4_block1_3_bn = model_to_prune.layers[index_conv4_block1_3_bn].output
            x = Add()([output_conv4_block1_0_bn, output_conv4_block1_3_bn])

    pruned_model = Model(inputs=model_prev.output, outputs=x)

    new_model = Sequential()
    new_model.add(model_prev)
    new_model.add(pruned_model)
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

def main():
    output_layer = 'conv4_block1_2_conv'
    model = load_model('../Models/NN/model_mnist_renet50_4_categories.h5')
    print_all_layers(model)
    resnet_model = model.get_layer('resnet50')

    feature_maps = extract_feature_maps(resnet_model, output_layer, np.random.rand(1, 224, 224, 3))

    num_samples, height, width, num_filters = feature_maps.shape
    feature_maps_flattened = feature_maps.reshape((num_samples, height * width, num_filters))

    # Reshape to (num_samples * height * width, num_filters) for clustering
    weights_list = feature_maps_flattened.reshape((num_filters, -1))

    pca = PCA(n_components=3)
    data = pca.fit_transform(weights_list)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x, h, s = complete_gradient_algorithm(data)

    x = scaler.fit_transform(x)
    z = cluster_algorithm(x, h, s, data)

    indexes = [random.choice(l) for l in z]

    prune_input = Input(shape=(32, 32, 3))

    pruned_model = prune_filter(resnet_model, output_layer, indexes)

    new_model = Sequential()
    new_model.add(prune_input)
    new_model.add(pruned_model)
    new_model.add(Dense(10, activation='softmax'))

    print_all_layers(new_model)

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = np.expand_dims(train_X, axis=-1)
    train_X = np.repeat(train_X, 3, axis=-1)
    train_X = np.pad(train_X, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
    train_y = to_categorical(train_y, 10)

    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    new_model.fit(x=train_X, y=train_y, epochs=1)

    test_X = np.expand_dims(test_X, axis=-1)
    test_X = np.repeat(test_X, 3, axis=-1)
    test_X = np.pad(test_X, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
    result = new_model.evaluate(test_X, to_categorical(test_y, 10))
    print(result)
    new_model.save('../NN/pruned_model_mnist_renet50_4_categories_1_layer.h5')

if __name__ == "__main__":
    main()