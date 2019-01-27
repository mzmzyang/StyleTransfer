import numpy as np
import scipy.io as sio
import tensorflow as tf
from PIL import Image

def vgg19(input_image):
    layers_name = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4','pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )

    vgg = sio.loadmat("imagenet-vgg-verydeep-19.mat")
    vgg_layers = vgg["layers"][0]

    layer_in = input_image
    network_model = {}

    for idx, name in enumerate(layers_name):
        layer_type = name[:4]
        if layer_type == 'conv':
            w, b = vgg_layers[idx][0][0][0][0]
            b = tf.constant(np.reshape(b,(b.size)))
            conv_out = tf.nn.conv2d(layer_in, w, strides=(1,1,1,1), padding='SAME') + b
            layer_out = tf.nn.relu(conv_out)
        elif layer_type == 'pool':
            layer_out = tf.nn.max_pool(layer_in, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
        network_model[name] = layer_out
        layer_in = layer_out

    return network_model

def loss_function(style_image,content_image,target_image):
    style_layers = [('conv1_2', 0.5), ('conv2_2', 1.0), ('conv3_2', 1.5), ('conv4_2', 3.0), ('conv5_2', 4.0)]
    content_layers = ['conv4_2']

    style_weight = 1000
    content_weight = 1
    loss = 0.0

    def content_loss(target_features, content_features):
        return tf.nn.l2_loss(target_features - content_features)

    def style_loss(target_features, style_features):
        _, height, width, channel = map(lambda i:i.value,target_features.get_shape())
        size = height * width * channel

        a = tf.reshape(target_features, (-1,channel))
        x = tf.reshape(style_features, (-1,channel))
        A = tf.matmul(tf.transpose(a), a)
        G = tf.matmul(tf.transpose(x), x)
        return tf.nn.l2_loss(G - A) / (size**2)


    style_features = vgg19([style_image])
    content_features = vgg19([content_image])
    target_features = vgg19([target_image])

    for layer in content_layers:
        loss += content_weight * content_loss(target_features[layer], content_features[layer])

    for layer, weight in style_layers:
        loss += style_weight * weight * style_loss(target_features[layer], style_features[layer])

    return loss

def style_transfer(style_image, content_image):
    style_input = tf.constant(style_image, dtype=tf.float32)
    content_input = tf.constant(content_image, dtype=tf.float32)
    target = tf.Variable(tf.random_normal(content_image.shape), dtype=tf.float32) * 0.4 + 0.6 * content_input

    total_loss = loss_function(style_input, content_input, target)
    train_op = tf.train.AdamOptimizer(1).minimize(total_loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(1000):
            _, loss, target_image = sess.run([train_op, total_loss, target])
            print("iter:%d, loss:%.9f" % (i, loss))
            if (i+1) % 100 == 0:
                image = np.clip(target_image + 128,0,255).astype(np.uint8)
                Image.fromarray(image).save("./img_%d.jpg" % (i+1))

if __name__ == '__main__':
    style = Image.open('style4.jpg')
    style = np.array(style).astype(np.float32) - 128.0
    content = Image.open('content.jpg')
    content = np.array(content).astype(np.float32) - 128.0
    style_transfer(style, content)