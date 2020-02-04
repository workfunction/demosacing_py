import os
import pickle

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model,load_model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import PReLU,Conv1D
from train.data import Data

batch_sz = 200
lr = 0.0001

class Train():
    def __init__(self, e_num=None, data=None, index=None):
        dirname = os.path.dirname(__file__)
        self.filepath = os.path.join(dirname, "model/my_model_1.h5")
        
        if e_num == None and os.path.isfile(self.filepath):
            self.model = tf.keras.models.load_model(self.filepath)
        else:
            assert type(data) is Data, "No model file or no input data"
            self.__test_train(e_num, data.train_label, data.train_data,
                              index, data.count)
    
    def __test_train(self, e_num, train_label, train_data, index, count):
        self.model = Sequential()
        self.model.add(Conv1D(1, (6), strides=1, padding='valid', 
                                activation='relu', use_bias=False,
                                kernel_initializer='glorot_uniform',
                                input_shape=(6,1)))
        
        self.model.compile(optimizer=tf.keras.optimizers.Nadam(lr), 
                    loss = 'mean_squared_error', metrics = ['mse'])
        
        train_x = train_data
        train_y = train_label

        checkpoint = ModelCheckpoint(self.filepath, monitor='mse', verbose=1, 
                                    save_best_only=True,
                                    save_weights_only=False, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(train_x, train_y, epochs=e_num, batch_size=batch_sz,
                       verbose=1, callbacks=callbacks_list)    
        self.model.summary()

    def look(self):        
        for layer in self.model.layers:        
            h=layer.get_weights()
            print(h[0])
            
            if len(h[0]) == 6:            
                for i in range(6):
                    print("*vc"+str(i)+" =",h[0][i][0][0],";")
                for i in range(6):
                    print("*vc"+str(i)+" =",h[0][5-i][0][0],";")
