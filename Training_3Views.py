import os
from keras import backend as K
# import keras as K
import sys
import numpy as np

from Library.Plot import draw_history
from Library.Network_Frame import ClassifierMPLSTM

train_folder = '/Train/CAM1'
val_folder= '/Val/CAM1'

input_path = os.path.join(os.path.dirname(__file__), 'Dataset')
output_path = os.path.join(os.path.dirname(__file__), 'Dataset/Model')

classifier = ClassifierMPLSTM()

his = classifier.fit(data_dir_path=input_path, model_folder=output_path, training_folder=train_folder, testing_folder=val_folder)
draw_history(his, ClassifierMPLSTM.LSTMmodel, output_path + '/accuracy.png')


if __name__ == '__main__':
    main()
