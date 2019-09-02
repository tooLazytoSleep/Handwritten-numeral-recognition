from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import matplotlib.pyplot as plt

# Define parameters
batch_size = 128
nb_classes = 2  # Classifier
nb_epoch = 7  # epoch

img_rows, img_cols = 28, 28  # image dimensions
nb_filters = 32  # number of filters
kernel_size = (3, 3)  # k-size
pool_size = (2, 2)  # pool_size

# Load data
(X_image, y_image), (X_validation, y_validation) = mnist.load_data()
#data preprocess
for i in range(len(y_image)):
    if y_image[i] % 2 == 0:
        y_image[i] = 1
    else:
        y_image[i] = 0
for i in range(len(y_validation)):
    if y_validation[i] % 2 == 0:
        y_validation[i] = 1
    else:
        y_validation[i] = 0
X_train = X_image[:55000]
X_test = X_image[55000:60000]
y_train = y_image[:55000]
y_test = y_image[55000:60000]

if K.image_dim_ordering() == 'th':
    # Using order from Theano：(conv_dim1,channels,conv_dim2,conv_dim3)
    X_train = X_train.reshape(55000, img_rows, img_cols, 1)
    X_test = X_test.reshape(5000, img_rows, img_cols, 1)
    X_validation = X_validation.reshape(10000, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
else:
    # Using order from Tensorflow：(conv_dim1,conv_dim2,conv_dim3,channels)
    X_train = X_train.reshape(55000, img_rows, img_cols, 1)
    X_test = X_test.reshape(5000, img_rows, img_cols, 1)
    X_validation = X_validation.reshape(10000, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# Normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_validation = X_validation.astype('float32')
X_train /= 255
X_test /= 255
X_validation /= 255

# Convert to Binary image
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_validation = np_utils.to_categorical(y_validation, nb_classes)


#Create Model
def create_model():
    # 2 convolution layers with pooling, a dropout layer, and 2 fully connected layers
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))

    # Compile the Model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#Train the Model
def train_model():
    model = create_model()
    result = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_validation, Y_validation))
    return result, model


#Model prediction and evaluation
def prediction():
    result, model = train_model()
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])  # 损失值  Test score: 0.0464644303082
    print('Test accuracy:', score[1])  # 准确率  Test accuracy: 0.9829
    print(result.history.keys())

    # Summarize history for loss
    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for accuracy
    plt.plot(result.history['acc'])
    plt.plot(result.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return model


#Save Model
def save_model():
    model = prediction()
    model.save("myModel.h5")


save_model()