import pathlib
import tensorflow as tf
import numpy as np
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_and_preprocess_from_path_label(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range
    return image, label


def loadDataSet():
    # 训练集文件夹，该文件夹下面是一系列标签文件夹，标签文件夹里面是图片
    data_path = pathlib.Path("./data_set/flower_data/train")
    # 读取标签文件夹下的所有jpg文件
    all_image_paths = list(data_path.glob('*/*'))
    # 所有图片路径的列表
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)

    # 读取文件夹标签
    label_names = sorted(item.name for item in data_path.glob('*/')
                         if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(
                          label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    # 切片
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths,
                                            all_image_labels))
    # 打包（图片，标签）
    image_label_ds = ds.map(load_and_preprocess_from_path_label)

    BATCH_SIZE = 32
    # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
    # 被充分打乱。
    ds = image_label_ds.shuffle(buffer_size=image_count)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return image_label_ds


if __name__ == '__main__':
    ds = loadDataSet()
    # vgg16(ds, 0.5, 5)
    print(ds)
