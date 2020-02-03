import numpy as np
import os
import pickle

import tensorflow as tf

from tensorflow.keras.models import Model,load_model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import PReLU,Conv1D
from input_data import Data

batch_sz = 200
lr = 0.0001

def test_train( e_num , train_label , train_data ,index,count):
    
    model = Sequential()
    model.add(Conv1D(1, (6), strides=1, padding='valid', 
                            activation='relu', use_bias=False,
                            kernel_initializer='glorot_uniform',
                            input_shape=(6,1)))
    
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), 
                  loss = 'mean_squared_error', metrics = ['mse'])
    
    train_x = train_data
    train_y = train_label

    filepath = "model/my_model_1.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='mse', verbose=1, 
                                 save_best_only=True,save_weights_only=False, 
                                 mode='min')
    callbacks_list = [checkpoint]
    model.fit(train_x, train_y, epochs=e_num, batch_size=batch_sz,verbose=1, 
              callbacks=callbacks_list)    
    model.summary()
    

def look():
    model = tf.keras.models.load_model("model/my_model_1.h5")
    
    for layer in model.layers:        
        h=layer.get_weights()
        print(h[0])
        
        if len(h[0]) == 6:            
            for i in range(6):
                print("*vc"+str(i)+" =",h[0][i][0][0],";")
            for i in range(6):
                print("*vc"+str(i)+" =",h[0][5-i][0][0],";")

if __name__ == "__main__":
    e_num = 20
    position_index = 0
    RGB_index = 1
    
    print("Loading data...")
    with open('data/data.pkl', 'rb') as input:
        data = pickle.load(input)

    test_train(e_num , data.train_label , data.train_data, position_index, data.count)
    look()
