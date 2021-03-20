import numpy as np
from keras import backend as K
import sys
import os
from keras_vggface.vggface import VGGFace
from keras.engine import  Model


def main():
    K.set_image_dim_ordering('tf')
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from Library.Network import ClassifierMPLSTM
    from Library.Loader import Load_TestList_and_labels

    data_folder = os.path.join(os.path.dirname(__file__), 'Dataset')    
    model_folder = os.path.join(os.path.dirname(__file__), 'Dataset','Model')
    
    path_to_conf_file = model_folder + '/' + ClassifierMPLSTM.LSTMmodel + '-config.npy'
    path_to_weight_file = model_folder + '/' + ClassifierMPLSTM.LSTMmodel + '-weights.h5'
    path_to_architecture_file = model_folder + '/' + ClassifierMPLSTM.LSTMmodel + '-architecture.json'

    test_path= '/Test/CAM1'

    predict = ClassifierMPLSTM()
    predict.load_model(path_to_conf_file, path_to_weight_file)

    videos = Load_TestList_and_labels(test_path, data_folder, [label for (label, label_index) in predict.label_list.items()])
    video_list = np.array([file_path for file_path in videos.keys()])

    correct_count = 0
    count = 0
    
    vgg_model = VGGFace(model='resnet50') 
    layer_name = 'avg_pool'
    out = vgg_model.get_layer(layer_name).output
    model = Model(vgg_model.input, out)
    
    correct_count = 0
    count = 0
    
    for v in video_list:
        
        label = videos[v]       
        predicted_label = predict.predict_lable(model, v)       
        correct_count = correct_count + 1 if label == predicted_label else correct_count
        accuracy = correct_count / (count+1)
        print('accuracy: ', accuracy)
        count += 1
        print(count)
    
if __name__ == '__main__':
    main()
