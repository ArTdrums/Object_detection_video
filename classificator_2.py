import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tensorflow import keras
from keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose, Concatenate, LeakyReLU, Dropout
import numpy as np

import random
namespace = {'NOTHING': 0, 'rule': 1 , 'shish': 2 }

fn = "D:\\images_6\\"
# формируем список всех xml файлов в папке
p = [fn + '/' + f for f in listdir(fn) if isfile(join(fn, f)) and f[-1] == 'l']


def load_img(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 256
    img = tf.image.resize(img, (1024, 1024))
    return img


# создаем запись
writer = tf.io.TFRecordWriter('classifier_rule_dataset.tfrecord')


def saveinrecord(img, name):
    # готовим данные, представляем в байтовом виде
    serialized_img = tf.io.serialize_tensor(img).numpy()
    serialized_name = tf.io.serialize_tensor(name).numpy()
    # собираем экзепмляр
    example = tf.train.Example(features=tf.train.Features(feature={
        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_img])),
        'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_name]))
    }))

    # записываем в запись
    writer.write(example.SerializeToString())


for xml in p:
    tree = ET.parse(xml)  # адрес файла
    root = tree.getroot()  # парсим
    num_objects = len(root) - 6

    w = int(root[4][0].text)  # ширина x
    h = int(root[4][1].text)  # высота y

    img = load_img(root[2].text)
    for num in range(num_objects):
        xmin = tf.clip_by_value(int(int(root[num + 6][4][0].text) / w * 1024), 0, 1024)
        ymin = tf.clip_by_value(int(int(root[num + 6][4][1].text) / h * 1024), 0, 1024)
        xmax = tf.clip_by_value(int(int(root[num + 6][4][2].text) / w * 1024), 0, 1024)
        ymax = tf.clip_by_value(int(int(root[num + 6][4][3].text) / h * 1024), 0, 1024)

        offset_height = ymin
        offset_width = xmin

        target_height = ymax - ymin
        target_width = xmax - xmin

        cropped = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
        cropped = tf.image.resize(cropped, (128, 128))

        name = namespace[root[num + 6][0].text]

        saveinrecord(cropped, name)

    # создадим и рамки фона
    counter = 0
    goal = 5

    while counter < goal:
        # сгенерим случайные координаты рамки
        gxmin = random.randint(1, 900)
        gymin = random.randint(1, 900)
        gxsize = random.randint(10, 100)
        gysize = random.randint(10, 100)

        gxmax = gxmin + gxsize
        gymax = gymin + gysize

        # а вдруг рамка пересекается с реальной?
        notintersect = True

        for num in range(num_objects):
            xmin = tf.clip_by_value(int(int(root[num + 6][4][0].text) / w * 1024), 0, 1024)
            ymin = tf.clip_by_value(int(int(root[num + 6][4][1].text) / h * 1024), 0, 1024)
            xmax = tf.clip_by_value(int(int(root[num + 6][4][2].text) / w * 1024), 0, 1024)
            ymax = tf.clip_by_value(int(int(root[num + 6][4][3].text) / h * 1024), 0, 1024)

            x_overlap = tf.maximum(0, tf.minimum(gxmax, xmax) - tf.maximum(gxmin, xmin))
            y_overlap = tf.maximum(0, tf.minimum(gymax, ymax) - tf.maximum(gymin, ymin))
            if x_overlap > 0 and y_overlap > 0:
                notintersect = False
                break

        if notintersect:
            cropped = tf.image.crop_to_bounding_box(img, gymin, gxmin, gysize, gxsize)
            cropped = tf.image.resize(cropped, (128, 128))
            name = 0
            saveinrecord(cropped, name)
            counter += 1

writer.close()

dataset = tf.data.TFRecordDataset('classifier_rule_dataset.tfrecord')


def parse_record(record):
    #нужно описать приходящий экземпляр
    #имена элементов как при записи
    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'name': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.io.parse_single_example(record, feature_description)
    img = tf.io.parse_tensor(parsed_record['img'], out_type=tf.float32)
    name = tf.io.parse_tensor(parsed_record['name'], out_type=tf.int32)
    return img, name

#пройдемся по записи и распакуем ее
dataset = dataset.map(parse_record)

#что-нибудь выведем
for i, c in dataset.take(1):
    print(i.shape)
    print(c)

dataset = tf.data.TFRecordDataset('classifier_rule_dataset.tfrecord')


def parse_record(record):
    # нужно описать приходящий экземпляр
    # имена элементов как при записи
    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'name': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.io.parse_single_example(record, feature_description)
    img = tf.io.parse_tensor(parsed_record['img'], out_type=tf.float32)
    name = tf.io.parse_tensor(parsed_record['name'], out_type=tf.int32)
    return img, name


# пройдемся по записи и распакуем ее
dataset = dataset.map(parse_record)

# еще раз проверим


dataset = dataset.shuffle(50).cache().prefetch(buffer_size=tf.data.AUTOTUNE).batch(64).shuffle(50)

for i, n in dataset.take(1):
    plt.figure(figsize=(10, 6))
    i = i.numpy()
    n = n.numpy()
    for nn in range(32):
        ax = plt.subplot(5, 10, 1 + nn)

        plt.title(n[nn])
        plt.imshow(i[nn])
        plt.axis('off')
    plt.show()

# создаем нейросеть классификатор
base_model = tf.keras.applications.MobileNetV2(weights='imagenet',
                                               include_top=False,
                                               input_shape=(128,128, 3))
base_model.trainable = False

inputs = Input((128,128,3))
x = base_model(inputs)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(3, activation = 'softmax')(x)

outputs = x

classifier = keras.Model(inputs, outputs)


class Model(tf.keras.Model):
    def __init__(self, nn):
        super(Model, self).__init__()
        self.nn = nn
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    def get_loss(self, y, preds):
        loss = tf.keras.losses.CategoricalCrossentropy()(tf.one_hot(y, 3), preds)
        return loss

    @tf.function
    def training_step(self, x, y):
        with tf.GradientTape() as tape:
            preds = self.nn(x)
            loss = self.get_loss(y, preds)

        gradients = tape.gradient(loss, self.nn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.nn.trainable_variables))
        return tf.reduce_mean(loss)


model = Model(classifier)
for i, c in dataset.take(1):
    print(tf.reduce_mean(model.training_step(i, c)))

from IPython.display import clear_output
hist = np.array(np.empty([0]))
epochs = 100

for epoch in range(1, epochs + 1):
    loss = 0
    lc = 0
    for step, (i, n) in enumerate(dataset):
        loss+=tf.reduce_mean(model.training_step(i,n))
        lc+=1
    clear_output(wait=True)
    print(epoch)
    hist = np.append(hist, loss/lc)
plt.plot(np.arange(0,len(hist)), hist)
plt.show()


def imshow_and_pred():
    n = 5
    plt.figure(figsize=(10, 6))
    for images, labels in dataset.take(1):
        preds = model.nn(images)
        for i in range(n):
            img = images[i]

            pred = preds[i].numpy()
            print(pred)
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(img, cmap='gist_gray')

            ma = pred.max()
            res = np.where(pred == ma)

            plt.title(str(res[0][0]) + ' ' + str(round(pred[res[0][0]], 3)))
            plt.axis('off')
            ax.get_yaxis().set_visible(False)
    plt.show()


imshow_and_pred()
model.nn.save('my_classifier.h5')