import cv2
import os
import numpy as np
import keras as keras
from keras_vggface.vggface import VGGFace
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.engine import  Model


def extract_test_features(model, video_path):

    if os.path.exists(video_path):
        feat_path=video_path.replace("CAM1", "CAM1-ResNet-Features-Test")
        feat_path=feat_path.replace("mp4", "npy")
        return np.load(feat_path)

    modelRes = VGGFace(model='resnet50')
    layer_name = 'avg_pool'
    out = modelRes.get_layer(layer_name).output
    model = Model(modelRes.input, out)
    
    print('Extracting ResNet features from: ', video_path)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    features = []
    success = True
        
    while success:
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        feature = model.predict(input).ravel()
        features.append(feature)
        success, image = vidcap.read()
    np_features1 = np.array(features)
    
    
    
    video_path=video_path.replace("CAM1", "CAM3")
    video_path=video_path.replace("v1", "v3")
    
    print('Extracting ResNet features from: ', video_path)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    features2 = []
    success = True
        
    while success:
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        feature = model.predict(input).ravel()
        features2.append(feature)
        success, image = vidcap.read()
    np_features2 = np.array(features2)


    video_path=video_path.replace("CAM3", "CAM5")
    video_path=video_path.replace("v3", "v5")
    
    print('Extracting ResNet features from: ', video_path)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    features3 = []
    success = True
        
    while success:
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        feature = model.predict(input).ravel()
        features3.append(feature)
        success, image = vidcap.read()
    np_features3 = np.array(features3)

    np_features=np.concatenate((np_features1, np_features2, np_features3), axis=1)
    
    return np_features



def extract_features(model, video_path, feature_path):
    if os.path.exists(feature_path):
        return np.load(feature_path)

    print('Extracting ResNet features from: ', video_path)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    features = []
    success = True
        
    while success:
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        feature = model.predict(input).ravel()
        features.append(feature)
        success, image = vidcap.read()
    np_features1 = np.array(features)
    
    
    
    video_path=video_path.replace("CAM1", "CAM3")
    video_path=video_path.replace("v1", "v3")
    
    print('Extracting ResNet features from: ', video_path)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    features2 = []
    success = True
        
    while success:
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        feature = model.predict(input).ravel()
        features2.append(feature)
        success, image = vidcap.read()
    np_features2 = np.array(features2)


    video_path=video_path.replace("CAM3", "CAM5")
    video_path=video_path.replace("v3", "v5")
    
    print('Extracting ResNet features from: ', video_path)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    features3 = []
    success = True
        
    while success:
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        feature = model.predict(input).ravel()
        features3.append(feature)
        success, image = vidcap.read()
    np_features3 = np.array(features3)

    np_features=np.concatenate((np_features1, np_features2, np_features3), axis=1)
    
       
    np.save(feature_path, np_features)
    return np_features


def extract_ResNet_features(dir_folder, output_folder, data_folder):


    input_path = dir_folder + '/' + data_folder
    output_path = dir_folder + '/' + output_folder


    model = VGGFace(model='resnet50')
    # model = VGGFace(model='vgg16')
    layer_name = 'avg_pool'
    # layer_name = 'fc7'
    out = model.get_layer(layer_name).output
    model_new = Model(model.input, out)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    labels = []
    samples = []

    count = 0
    for x in os.listdir(input_path):
        
        file_path = input_path + os.path.sep + x
        if not os.path.isfile(file_path):
            class_folder = x
            output_class_path = output_path + os.path.sep + class_folder
            if not os.path.exists(output_class_path):
                os.makedirs(output_class_path)
            count += 1
            coun=0
            for xx in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + xx
                feat_file_path = output_class_path + os.path.sep + xx.split('.')[0] + '.npy'
                feat = extract_features(model_new, video_file_path, feat_file_path)
                labels.append(x)
                samples.append(feat)


    return samples, labels

