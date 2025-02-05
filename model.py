import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense
from keras.preprocessing.image import load_img, img_to_array
from data import split_data
from keras.utils import to_categorical

os.chdir(r"/home/husammm/Desktop/Courses/Python/MLCourse/uni/ECG_QRS/database/")


def load_images(directory_path, heightOfPicture, widthOfPicture):
    image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.jpg')]
    
    images = []
    for path in image_paths:
        # Load and preprocess each image
        img = load_img(path, target_size=(heightOfPicture, widthOfPicture, 3))
        img_array = img_to_array(img) / 255.0 
        images.append(img_array)
    
    return images

def load_label_data(heightOfPicture, widthOfPicture):
    Qrs_imgaes = load_images('QRS/', heightOfPicture, widthOfPicture)
    notQrs_imgaes = load_images('notQRS/', heightOfPicture, widthOfPicture)

    Qrs_imgaes = np.array(Qrs_imgaes)
    notQrs_imgaes = np.array(notQrs_imgaes)

    Qrs_labels = np.ones(len(Qrs_imgaes))
    notQrs_labels = np.zeros(len(notQrs_imgaes))

    images = np.concatenate([Qrs_imgaes, notQrs_imgaes])
    labels = np.concatenate([Qrs_labels, notQrs_labels])
    

    x_train, x_test, t_train, t_test = split_data(images, labels)
    x_train, x_val, t_train, t_val = split_data(x_train, t_train, split = 0.1)

    return x_train, x_val, x_test, t_train, t_val, t_test

def build_model(dimentions, filters, kernel_size, optimizer):
    heightOfPicture = dimentions[0]
    widthOfPicture = dimentions[1]

    model = Sequential()
    model.add(Conv2D(filters=filters,kernel_size=kernel_size, input_shape=(heightOfPicture,widthOfPicture,3),activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=kernel_size, input_shape=(heightOfPicture,widthOfPicture,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Conv2D(filters=128,kernel_size=(3,3), input_shape=(heightOfPicture,widthOfPicture,3),activation='relu'))
    model.add(Flatten())

    model.add(Dense(64,activation='relu'))
    # model.add(Dense(10,activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])

    return model

def train_and_evaluate_model(model, x_train, t_train, x_val, t_val, x_test, t_test, check_point, threshold = 0.5):
    # check_point = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    model.fit(x_train, t_train, epochs=10, batch_size=32, validation_data=(x_val, t_val), callbacks= check_point)
    predictions = model.predict(x_test)
    binary_preds = (predictions > threshold).astype(int)
    test_loss, test_accuracy = model.evaluate(x_test, t_test)
    return binary_preds, test_loss, test_accuracy
