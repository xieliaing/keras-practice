import numpy as np
import io, gzip, requests, gc
import keras
import keras.backend as K

from keras.models import Model, Sequential
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Flatten, Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
import cv2
import h5py as h5py

from keras.callbacks import EarlyStopping
np.random.seed(962342)

train_image_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
train_label_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
test_image_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
test_label_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"

train_image_url = ".\\data\\train-images-idx3-ubyte.gz"
train_label_url = ".\\data\\train-labels-idx1-ubyte.gz"
test_image_url = ".\\data\\t10k-images-idx3-ubyte.gz"
test_label_url = ".\\data\\t10k-labels-idx1-ubyte.gz"

def readRemoteGZipFile(url, isLabel=True):
    response=requests.get(url, stream=True)
    gzip_content = response.content
    fObj = io.BytesIO(gzip_content)
    content = gzip.GzipFile(fileobj=fObj).read()
    if isLabel:
        offset=8
    else:
        offset=16
    result = np.frombuffer(content, dtype=np.uint8, offset=offset)    
    return(result)

def readLocalGZipFile(file, isLabel=True):
    buffer = gzip.open(file, "rb").read()
    if isLabel:
        offset=8
    else:
        offset=16
    result = np.frombuffer(buffer, dtype=np.uint8, offset=offset)  
    return(result)

train_labels = readLocalGZipFile(train_label_url, isLabel=True)
train_images_raw = readLocalGZipFile(train_image_url, isLabel=False)

test_labels = readLocalGZipFile(test_label_url, isLabel=True)
test_images_raw = readLocalGZipFile(test_image_url, isLabel=False)

'''
train_labels = readRemoteGZipFile(train_label_url, isLabel=True)
train_images_raw = readRemoteGZipFile(train_image_url, isLabel=False)

test_labels = readRemoteGZipFile(test_label_url, isLabel=True)
test_images_raw = readRemoteGZipFile(test_image_url, isLabel=False)
'''

train_images = train_images_raw.reshape(len(train_labels), 784)
test_images = test_images_raw.reshape(len(test_labels), 784)

X_train = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
X_test = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

X_train /= 255
X_test /= 255

X_train -= 0.5
X_test -= 0.5

X_train *= 2.
X_test *= 2.

Y_train = train_labels
Y_test = test_labels
Y_train2 = keras.utils.to_categorical(Y_train).astype('float32')
Y_test2 = keras.utils.to_categorical(Y_test).astype('float32')

from sklearn import manifold
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
plt.rcParams['figure.figsize']=(20, 10)
# Scale and visualize the embedding vectors
def plot_embedding(X, Image, Y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(Y[i]),
                 color=plt.cm.Set1(Y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(Image[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

'''
samples=np.random.choice(range(len(Y_train)), size=1500)
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
sample_images = train_images[samples]
sample_targets = train_labels[samples]
X_tsne = tsne.fit_transform(sample_images)
t1 = time()

plot_embedding(X_tsne, sample_images.reshape(sample_targets.shape[0], 28, 28), sample_targets,
               "t-SNE embedding of the digits (time %.2fs)" %
               (t1 - t0))
plt.savefig('./tSNE.png')
plt.close()
'''
num_classes = len(set(Y_train))
model3=Sequential()
model3.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", 
          input_shape=X_train.shape[1:], activation='relu'))
model3.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'))
model3.add(MaxPooling2D(pool_size=(3, 3)))
model3.add(Dropout(0.5))
model3.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))
model3.add(Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation='relu'))
model3.add(MaxPooling2D(pool_size=(3, 3)))
model3.add(Dropout(0.5))
model3.add(Flatten())
model3.add(Dense(256))
model3.add(LeakyReLU())
model3.add(Dropout(0.5))
model3.add(Dense(256))
model3.add(LeakyReLU())
#model2.add(Dropout(0.5))
model3.add(Dense(num_classes, activation='softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.summary()

model3fit = model3.fit(X_train, Y_train2, validation_data=(X_test, Y_test2), batch_size=500, verbose=1, epochs=100)

score = model3.evaluate(X_test, Y_test2)
print(score)