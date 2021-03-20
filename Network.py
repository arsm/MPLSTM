import os
import math
import numpy as np
from keras import backend as K
# import keras as K
from keras.layers import  Input, Dense, Activation, Dropout, Layer, Bidirectional, Flatten, Conv2D, Reshape, Concatenate
from Library.MPLSTM_3inputs import LSTM
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils, plot_model
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask



from Library.CNNfeature_extractor import extract_ResNet_features, extract_test_features

batch_size = 60
no_of_epochs = 500
hidden_layer_size=256


# K.set_image_dim_ordering('tf')
K.image_data_format()




def batch_generation(Bsamples, Blabels):
    num_batches = len(Bsamples) // batch_size

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = (batchIdx + 1) * batch_size
            yield np.array(Bsamples[start:end]), Blabels[start:end]


def batch_generation2in(Bsamples, Blabels):
    num_batches = len(Bsamples) // batch_size

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = (batchIdx + 1) * batch_size
            yield [np.array(Bsamples[start:end]),np.array(Bsamples[start:end])], Blabels[start:end]
            
            
        


class ClassifierMPLSTM(object):
    LSTMmodel = 'MP-lstm'
    
    def __init__(self):
        self.model = None
        self.config = None
        self.label_list = None
        self.labels_index = None
        self.feat_size = None
        self.no_of_class = None
        self.no_frames = None
        

    def attention_3d_block(self, inputs):
        from keras.layers import Permute, Reshape, Lambda, RepeatVector, Multiply
        
        a = Permute((2, 1))(inputs)
        a = Dense(self.no_frames, activation='softmax')(a)

        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul
    
    

    def create_model(self):

        # num_capsule=6
        # dim_capsule=64
        # routings=3
        
        inputs = Input(shape=(self.no_frames, self.feat_size))             
        x = Bidirectional(LSTM(units=hidden_layer_size, return_sequences=True, dropout=0.20))(inputs)
        x = self.attention_3d_block(x)
        # x= Reshape((-1,16, 16))(x)
        # x = PrimaryCap(x, dim_capsule, n_channels=5, kernel_size=3, strides=1, padding='valid')

        x = Flatten()(x)
        predictions = Dense(self.no_of_class, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  
        model.summary()
        
        return model


    def create_model2in(self):

       
        inputs = Input(shape=(self.no_frames, self.feat_size))         
        x = LSTM(units=hidden_layer_size, return_sequences=True, dropout=0.20)(inputs)
        # x = self.attention_3d_block(x)
        x = Model(inputs=inputs, outputs=x)
        
        inputs2 = Input(shape=(self.no_frames, self.feat_size)) 
        y = LSTM(units=hidden_layer_size, return_sequences=True, dropout=0.20)(inputs2)
        # y = self.attention_3d_block(y)
        y = Model(inputs=inputs2, outputs=y)
        
        combined = Concatenate(2)([x.output, y.output])
        z = Flatten()(combined)
        predictions = Dense(self.no_of_class, activation='softmax')(z)
        
        model = Model(inputs=[x.input, y.input], outputs=predictions)
        
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  
        return model

    
    

    def load_model(self, path_to_conf_file, path_to_weight_file):

        config = np.load(path_to_conf_file).item()
        self.feat_size = config['feat_size']
        self.no_of_class = config['no_of_class']
        self.label_list = config['label_list']
        self.no_frames = config['no_frames']
        self.labels_index = dict([(ind, txt) for txt, ind in self.label_list.items()])
        self.model = self.create_model()
        self.model.load_weights(path_to_weight_file)


    def predict_lable(self, model, video_path):
        x = extract_test_features(model, video_path)
        frames = x.shape[0]
        if frames > self.no_frames:
            x = x[0:self.no_frames, :]
        elif frames < self.no_frames:
            temp = np.zeros(shape=(self.no_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_class = np.argmax(self.model.predict(np.array([x]))[0])
        predicted_label = self.labels_index[predicted_class]
        return predicted_label

    
    def fit(self, data_dir_path, model_folder, training_folder='Train/CAM1', testing_folder='Test/CAM1'):
        
        path_to_conf_file = model_folder + '/' + ClassifierMPLSTM.LSTMmodel + '-config.npy'
        path_to_weight_file = model_folder + '/' + ClassifierMPLSTM.LSTMmodel + '-weights.h5'
        path_to_architecture_file = model_folder + '/' + ClassifierMPLSTM.LSTMmodel + '-architecture.json'

        training_folder_feat = training_folder + '-ResNet-Features-Train'
        testing_folder_feat = testing_folder + '-ResNet-Features-Validation'
        
        
        self.label_list = dict()
                
        training_samples, training_labels = extract_ResNet_features(data_dir_path,
                                                                output_folder=training_folder_feat,
                                                                data_folder=training_folder)

        testing_samples, testing_labels = extract_ResNet_features(data_dir_path,
                                                                  output_folder=testing_folder_feat,
                                                                  data_folder=testing_folder)
            
        self.feat_size = training_samples[0].shape[1]
        fr_list = []
        max_fr = 0

        for x in training_samples:
            fr = x.shape[0]
            fr_list.append(fr)
            max_fr = max(fr, max_fr)
            self.no_frames = int(np.mean(fr_list))
        
        print('mean number of frames: ', self.no_frames)
        print('maximum number of frames: ', max_fr)
        self.no_frames = 20
        
        for i in range(len(training_samples)):
            x = training_samples[i]
            frames = x.shape[0]
            print(x.shape)
            if frames > self.no_frames:
                x = x[0:self.no_frames, :]
                training_samples[i] = x
            elif frames < self.no_frames:
                temp = np.zeros(shape=(self.no_frames, x.shape[1]))
                temp[0:frames, :] = x
                training_samples[i] = temp
        for y in training_labels:
            if y not in self.label_list:
                self.label_list[y] = len(self.label_list)
        print(self.label_list)
        for i in range(len(training_labels)):
            training_labels[i] = self.label_list[training_labels[i]]

        self.no_of_class = len(self.label_list)
        training_labels = np_utils.to_categorical(training_labels, self.no_of_class)


        for i in range(len(testing_samples)):
            x = testing_samples[i]
            frames = x.shape[0]
            print(x.shape)
            if frames > self.no_frames:
                x = x[0:self.no_frames, :]
                testing_samples[i] = x
            elif frames < self.no_frames:
                temp = np.zeros(shape=(self.no_frames, x.shape[1]))
                temp[0:frames, :] = x
                testing_samples[i] = temp
        for y in testing_labels:
            if y not in self.label_list:
                self.label_list[y] = len(self.label_list)
        print(self.label_list)
        for i in range(len(testing_labels)):
            testing_labels[i] = self.label_list[testing_labels[i]]

        self.no_of_class = len(self.label_list)

        testing_labels = np_utils.to_categorical(testing_labels, self.no_of_class)
        
        

        config = dict()
        config['label_list'] = self.label_list
        config['no_of_class'] = self.no_of_class
        config['feat_size'] = self.feat_size
        config['no_frames'] = self.no_frames
        self.config = config

        np.save(path_to_conf_file, config)

        model = self.create_model()
        # plot_model(model, "my_first_model.png")
        open(path_to_architecture_file, 'w').write(model.to_json())


        train_gen = batch_generation(training_samples, training_labels)
        test_gen = batch_generation(testing_samples, testing_labels)
        
        num_train_batches = len(training_samples) // batch_size
        num_test_batches = len(testing_samples) // batch_size
        
        checkpoint = ModelCheckpoint(filepath=path_to_weight_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        hist = model.fit_generator(generator=train_gen, steps_per_epoch=num_train_batches,
                                      epochs=no_of_epochs, 
                                      verbose=1, validation_data=test_gen, validation_steps=num_test_batches,
                                      callbacks=[checkpoint])
        
        
        return hist
