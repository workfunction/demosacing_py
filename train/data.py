import os
import cv2
import pickle

import numpy as np

from PIL import Image

class Data():
    def __init__(self, index=None, RGB_index=None):
        self.dirname = os.path.dirname(__file__)
        datafile = os.path.join(self.dirname, "data/data.pkl")
        if RGB_index == None and os.path.isfile(datafile):
            print("Loading data...")
            with open(datafile, 'rb') as input:
                dict = pickle.load(input)
            self.__dict__.update(dict)
        else:
            self.make_data(index, RGB_index)

    def make_data(self, index, RGB_index):
        origin_path = os.path.join(self.dirname, "T91/origin_bak/")
        cv_path = os.path.join(self.dirname, "T91/cv_0.5_bak/")
        
        in_file_name = []
        root_path = origin_path
        for root_path, dir_name, file_name in os.walk(root_path):
            for f in file_name:    
                in_file_name.append(f)
            
        offset = 7#這個有改紀錄一下
        size_sum = 0
        for i in range(len(in_file_name)):
            p = origin_path + in_file_name[i] 
            #print(p)
            im = Image.open(p)
            w = im.size[0]
            h = im.size[1]
            if (w % 2) == 1:
                w = w - 1
            if (h % 2) == 1:
                h = h - 1
            im1 = im.crop((0, 0, w, h))        
            
            new_dimension = (int(w/2),int(h/2))        
            im_05 = im1.resize(new_dimension, Image.BICUBIC)
            im_05.save(os.path.join(cv_path + in_file_name[i]))
        
            
            size = int(((h-offset*2)*(w-offset*2))/4)
            size_sum = size_sum + size
        
        map_size = 1
        scale_x = 1
        scale_y = 6
        train_label = np.zeros((size_sum,1),dtype="float32")
        train_data = np.zeros((size_sum,map_size*scale_x,scale_y),dtype="float32")
        
        print("making training data...")
        count = 0
        
        
        for i in range(len(in_file_name)):
            im = cv2.imread(origin_path + in_file_name[i])
            im_05 = cv2.imread(cv_path + in_file_name[i])
            
        
            w = im.shape[1]
            h = im.shape[0]
            if (w % 2) == 1:
                w = w - 1
            if (h % 2) == 1:
                h = h - 1
            
            print(in_file_name[i] + " finished")
        
            for i in range(0+offset,h-offset,2):
                for j in range(0+offset,w-offset,2):
                #train_label[count] = im[i,j,1] / 255
                    train_label[count,0] = im[i,j,RGB_index]
                    #d = int((scale / 2)-1)
                    index_i = int((i - 1)/2) - 2
                    #index_i = int(i)
                    index_j = int((j - 1)/2) - 2 
                    
                    for k in range(0,scale_y):
                        train_data[count,0,k] = (
                            im_05[index_i,index_j+k,RGB_index]*0.09423056 + 
                            im_05[index_i+1,index_j+k,RGB_index]*(-0.29461205) + 
                            im_05[index_i+2,index_j+k,RGB_index]*(1.0735404) + 
                            im_05[index_i+3,index_j+k,RGB_index]*(0.16020575) + 
                            im_05[index_i+4,index_j+k,RGB_index]*(-0.044060647) + 
                            im_05[index_i+5,index_j+k,RGB_index]*(0.01062963))
                                
                    #d1 = train_data[count,0,3] - train_data[count,0,2]
                    #if(d1 < 16 and d1 > 4):
                    #print(i,j,train_label[count])
                    count = count + 1
                    #print(count)
        
        train_data = train_data.reshape(size_sum,map_size*scale_x*scale_y,1)
        train_label = train_label.reshape(size_sum,1,1)
        train_data = train_data[0:count,:,:]
        train_label = train_label[0:count,:,:]
        
        print("training data size : " + str(train_data.shape))
        print("training label size : " + str(train_label.shape))
        
        self.train_label = train_label
        self.train_data = train_data
        self.size = size_sum
        self.count = count
    
    def save(self):
        print("Saving data...")
        with open(os.path.join(self.dirname, "data/data.pkl"), "wb") as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)
