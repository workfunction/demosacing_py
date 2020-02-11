import os
import pickle

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from numba import int64
from numba import float32
from PIL import Image
from lib.mosaic import Mosaic
from lib.mepi_delta import MEPR

spec = [
    ('count0', int64),
    ('count1', int64),
    ('train_label1', float32[:, :, :]),
    ('train_label0', float32[:, :, :]),
    ('train_data1', float32[:, :, :]),
    ('train_data0', float32[:, :, :]),
]

#@nb.jitclass(spec)  
class Data():
    def __init__(self, index=None):
        self.dirname = os.path.dirname(__file__)
        datafile = os.path.join(self.dirname, "data/data.pkl")
        if index == None and os.path.isfile(datafile):
            print("Loading data...")
            with open(datafile, 'rb') as input:
                dict = pickle.load(input)
            self.__dict__.update(dict)
        else:
            self.count0 = 0
            self.count1 = 0
            self.make_data(index)

    def make_data(self, index):
        origin_path = os.path.join(os.path.dirname(__file__), "T91/origin_bak/")
        cv_path = os.path.join(os.path.dirname(__file__), "T91/cv_0.5_bak/")
        
        in_file_name = []
        root_path = origin_path
        for root_path, dir_name, file_name in os.walk(root_path):
            for f in file_name:
                in_file_name.append(f)
            
        offset = 11#這個有改紀錄一下
        RGB_index = 1
        
        scale_y = 6
        
        print("making training data...")
        count = 0
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
                
            size = int(((h-offset*2)*(w-offset*2))/4)
            size_sum = size_sum + size
        
        data_sum = int(size_sum/2)
        train_label0 = np.zeros((data_sum+1, 1, 1), dtype="float32")
        train_data0 = np.zeros((data_sum+1, scale_y, 1), dtype="float32")
        train_label1 = np.zeros((data_sum, 1, 1), dtype="float32")
        train_data1 = np.zeros((data_sum, scale_y, 1), dtype="float32")
        
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
            iarray = np.array(im1, dtype=np.uint8)
            
            new_dimension = (int(w/2),int(h/2))        
            im_05 = im1.resize(new_dimension, Image.BICUBIC)
            array = np.array(im_05, dtype=np.uint8)
            iarray_05 = Mosaic(array, im_05.width, im_05.height).Algorithm()
            
            delta = MEPR(iarray_05, 2).Algorithm()
            
            image = Image.fromarray(iarray_05)
            image.save(os.path.join(cv_path + in_file_name[i]))
            
            image = Image.fromarray(delta[:, :, 0])
            image.save(os.path.join(cv_path + "d_" +in_file_name[i]))
            
            #plt.imshow(iarray_05, cmap='gray', vmin=0, vmax=255)
            #plt.show()
            #plt.imshow(iarray[:, :, 1], cmap='gray', vmin=0, vmax=255)
            #plt.show()
            
            print(in_file_name[i] + " finished")
        
            for i in range(0+offset, h-offset, 2):
                for j in range(0+offset, w-offset, 2):
                    label = np.zeros((1, 1), dtype="float32")
                    data = np.zeros((scale_y, 1), dtype="float32")
                    
                    label[0, 0] = iarray[i, j, 1]

                    #index_j = int((j + 0.5) * (0.5) - 0.5)
                    #index_i = int((i + 0.5) * (0.5) - 0.5)
                    index_i = int((i - 1)/2)
                    index_j = int((j - 1)/2)
                    
                    data[0, 0] = delta[i, j, 0]
                    data[1, 0] = delta[i, j, 1]
                    data[2, 0] = iarray_05[index_i, index_j]
                    data[3, 0] = iarray_05[index_i+1, index_j]
                    data[4, 0] = iarray_05[index_i, index_j+1]
                    data[5, 0] = iarray_05[index_i+1, index_j+1]
                    
                    if count % 2 == 0:
                        train_label0[self.count0, :, :] = label[:, :]
                        train_data0[self.count0, :, :] = data[:, :]
                        self.count0 = self.count0 + 1
                    else:
                        train_label1[self.count1, :, :] = label[:, :]
                        train_data1[self.count1, :, :] = data[:, :]
                        self.count1 = self.count1 + 1
                                
                    #d1 = train_data[count,0,3] - train_data[count,0,2]
                    #if(d1 < 16 and d1 > 4):
                    #print(i,j,train_label[count])
                    count = count + 1
                    #print(count)
                    
        self.train_label0 = train_label0
        self.train_data0  = train_data0 
        self.train_label1 = train_label1
        self.train_data1  = train_data1 
        
        print("training data size : " + str(self.train_data0.shape))
        print("training label size : " + str(self.train_label0.shape))
        print("training size : " + str(self.count0))
        print(i)
        print(j)
        print(index_i)
        print(index_j)
    
    def save(self):
        print("Saving data...")
        with open(os.path.join(self.dirname, "data/data.pkl"), "wb") as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)
