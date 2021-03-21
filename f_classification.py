import os
import cv2
import random
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.models import Model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def read_img(path):
    data = []
    labels = []
    for name in os.listdir(path):
        flowers = os.listdir(path + '/' + name)
        for flower in flowers:
            # 图片的每一个像素点变成RGB值
            img = cv2.imread(path + '/' + name + '/' + flower)
            img = cv2.resize(img, (224, 224))
            data.append(img)
            labels.append(name)
    return np.asarray(data), np.asarray(labels)


def get_data(data, label):
    # 打乱
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    # 修改label
    type = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3,
            "tulips": 4}
    labels = np.zeros((len(label), 5))
    for i in range(len(labels)):
        labels[i][type[label[i]]] = 1
    return np.array(data), np.array(labels)


def vgg16(train_data, train_labels, val_data, val_labels):
    # 使用keras内置的vgg16
    # weights：指定模型初始化的权重检查点、
    # include_top: 指定模型最后是否包含密集连接分类器。
    # 默认情况下，这个密集连接分类器对应于ImageNet的100个类别。
    # 如果打算使用自己的密集连接分类器，可以不适用它，置为False。
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model_vgg16 = VGG16(weights='imagenet', include_top=False,
                        input_shape=(224, 224, 3))
    # 查看model_vgg16的架构
    # model_vgg16.summary()
    # 权值冻结
    for layer in model_vgg16.layers:
        layer.trainable = False
    # 用于将输入层的数据压成一维的数据
    model = layers.Flatten(name='flatten')(model_vgg16.output)
    # 全连接层，输入的维度z
    model = layers.Dense(64, activation='relu')(model)
    # 每一批的前一层的激活值重新规范化，输出均值接近0，标准差接近1
    model = layers.BatchNormalization()(model)
    model = layers.Dropout(0.5)(model)
    model = layers.Dense(32, activation='relu')(model)
    model = layers.BatchNormalization()(model)
    model = layers.Dropout(0.5)(model)
    model = layers.Dense(16, activation='relu')(model)
    model = layers.BatchNormalization()(model)
    model = layers.Dropout(0.5)(model)
    model = layers.Dense(5, activation='softmax')(model)
    model = Model(inputs=model_vgg16.input, outputs=model, name='vgg16')

    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    model.fit(train_data, train_labels, batch_size=32, epochs=50,
              validation_data=(val_data, val_labels))
    # return history


if __name__ == '__main__':
    train_path = 'data_set/flower_data/train'
    val_path = 'data_set/flower_data/val'

    train_data, train_labels = read_img(train_path)
    val_data, val_labels = read_img(val_path)

    train_data, train_labels = get_data(train_data, train_labels)
    val_data, val_labels = get_data(val_data, val_labels)

    vgg16(train_data, train_labels, val_data, val_labels)
