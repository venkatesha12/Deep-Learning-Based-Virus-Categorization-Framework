root = "dataRaw"
import sys
import os
from math import log
import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

num_classes = 25


## This function allows us to process our hexadecimal files into png images##
def convertAndSave(array, name):
    print('Processing ' + name)
    if array.shape[1] != 16:  # If not hexadecimal
        assert (False)
    b = int((array.shape[0] * 16) ** (0.5))
    b = 2 ** (int(log(b) / log(2)) + 1)
    a = int(array.shape[0] * 16 / b)
    array = array[:a * b // 16, :]
    array = np.reshape(array, (a, b))
    im = Image.fromarray(np.uint8(array))
    im.save(root + '\\' + name + '.png', "PNG")
    return im


# plots images with labels within jupyter notebook
def plots(ims, figsize=(20, 30), rows=10, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = 10  # len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(0, 50):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(list(batches.class_indices.keys())[np.argmax(titles[i])], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


def malware_model():
    Malware_model = Sequential()
    Malware_model.add(Conv2D(30, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=(64, 64, 3)))

    Malware_model.add(MaxPooling2D(pool_size=(2, 2)))
    Malware_model.add(Conv2D(15, (3, 3), activation='relu'))
    Malware_model.add(MaxPooling2D(pool_size=(2, 2)))
    Malware_model.add(Dropout(0.25))
    Malware_model.add(Flatten())
    Malware_model.add(Dense(128, activation='relu'))
    Malware_model.add(Dropout(0.5))
    Malware_model.add(Dense(50, activation='relu'))
    Malware_model.add(Dense(num_classes, activation='softmax'))
    Malware_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return Malware_model


def main():
    # Get the list of files
    files = os.listdir(root)
    print('files : ', files)
    # We will process files one by one.
    for counter, name in enumerate(files):
        # We only process .bytes files from our folder.
        if '.bytes' != name[-6:]:
            continue
        f = open(root + '/' + name)
        array = []
        for line in f:
            xx = line.split()
            if len(xx) != 17:
                continue
            array.append([int(i, 16) if i != '??' else 0 for i in xx[1:]])
        # plt.imshow(convertAndSave(np.array(array),name))
        del array
        f.close()

    path_root = "malimg_paper_dataset_imgs\\"

    from keras.preprocessing.image import ImageDataGenerator
    batches = ImageDataGenerator().flow_from_directory(directory=path_root, target_size=(64, 64), batch_size=10000)
    print(batches.class_indices)
    imgs, labels = next(batches)
    print(imgs.shape)
    print(labels.shape)
    classes = batches.class_indices.keys()
    perc = (sum(labels) / labels.shape[0]) * 100
    plt.xticks(rotation='vertical')
    plt.bar(classes, perc)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(imgs / 255., labels, test_size=0.3)
    print(X_train.shape)

    Malware_model = malware_model()
    y_train_new = np.argmax(y_train, axis=1)

    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train_new),
                                                      y_train_new)
    Malware_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, class_weight=class_weights)
    Malware_model.save('malware_cnn.h5')

    scores = Malware_model.evaluate(X_test, y_test)

    print('Final CNN accuracy: ', scores[1])
    return scores


if __name__ == '__main__':
    main()
