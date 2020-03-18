#! /usr/bin/env python

import sys
import os
import math
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from PIL import Image
from lib.mepi import MEPI
from lib.mepi_legacy import MEPL
from lib.mosaic import Mosaic
from lib.MGBI5 import MGBI_5
from lib.mgepi import MGEPI

def read_img(img):
	return tf.convert_to_tensor(img, dtype=np.uint8)

@tf.function
def do_psnr(tf_img1, tf_img2):
	return tf.image.psnr(tf_img1, tf_img2, max_val=255)

def run(path):
    global config
    
    oim = Image.open(path)

    w = oim.size[0]
    h = oim.size[1]
    if (w % 2) == 1:
        w = w - 1
    if (h % 2) == 1:
        h = h - 1
    im1 = oim.crop((0, 0, w, h))        
    oimage = np.array(im1, dtype=np.uint8)
    
    new_dimension = (int(w/2), int(h/2))        
    im = im1.resize(new_dimension, Image.BICUBIC)
    image = np.array(im, dtype=np.uint8)

    mos = Mosaic(image, im.width, im.height)
    mimage = mos.Algorithm()

    if config["DELT"] == True:
        epi = MEPI(mimage, 2)
        out = epi.Delta()
        config["SHOW"] = True
        config["SHOWG"] = True
        config["SAVE"] = False
    else:
        epi = MEPI(mimage, 2)
        out = epi.Algorithm()
    
    if config["CROP"] == False:
        #補原圖
        out[:config["EDGE"], :, :] = oimage[:config["EDGE"], :, :]
        out[:, :config["EDGE"], :] = oimage[:, :config["EDGE"], :]
        out[-1*config["EDGE"]:, :, :] = oimage[-1*config["EDGE"]:, :, :]
        out[:, -1*config["EDGE"]:, :] = oimage[:, -1*config["EDGE"]:, :]
    else:
        #裁減
        out = out[config["EDGE"]:-1*config["EDGE"], config["EDGE"]:-1*config["EDGE"], :]
        oimage = oimage[config["EDGE"]:-1*config["EDGE"], config["EDGE"]:-1*config["EDGE"], :]

    if config["SHOW"] == True:
        if config["SHOWG"] == True:
            if config["DELT"] == True:
                plt.imshow(out[...,0], cmap='hsv', vmin=-255, vmax=255)
            else:
                plt.imshow(out[...,1], cmap='hsv', vmin=0, vmax=255)
            plt.show()
            assert False
        else:
            plt.imshow(out)
        plt.show()

    filename, ext = os.path.splitext(path)
    if config["SAVE"] == True:
        im = Image.fromarray(out)
        im.save("result/" + os.path.basename(filename) + config["POST"] + ext)

    p = float(do_psnr(read_img(oimage), read_img(out)))
    s = os.path.basename(filename) + ": " + str(p)
    print(s)
    return [p, (s + "\n")]

def main():
    global config
    if len(sys.argv) != 3:
        print("[ARGS] No input file!")
        return -1
    arg = sys.argv[1]

    if os.path.isfile(arg):
        files = [arg]
    elif os.path.isdir(arg):
        files = [os.path.join(arg, f) for f in os.listdir(arg)]
    else:
        print("[ARGS] File '" + arg + "' does not exist!")
        return -1
    
    if not os.path.isfile('config.json'):
        print("[CONF] No config file! Saving default config...")
        print("")
        config = {"CROP" : True,
                  "EDGE" : 10,
                  "SAVE" : True,
                  "SHOW" : True,
                  "SHOWG": False,
                  "POST" : "_2x",
                  "DELT" : False}
        
        with open('config.json', 'w') as f:
            f.writelines(json.dumps(config, sort_keys = True, indent = 4))            
    
    else:
        with open('config.json', 'r') as f:
            config = json.load(f)
    
    num_cores = mp.cpu_count()
    if len(files) > 1:
        config["SHOW"] = False
        config["DELT"] = False
        
        files.sort()

    print("Files to be run:")
    print(*files, sep="\n")
    print("============================")
    
    pool = mp.Pool(processes=(num_cores if len(files) > num_cores else len(files)))
    ss = np.array(pool.map(run, files)) 
    pool.close()  
    pool.join()
    
    f = open("result/demofile2.txt", "w")
    f.writelines(ss[:,1])
    f.close()
    n = ss[:,0].astype("float")
    print("AVG = " + str(np.average(n)))

if __name__ == "__main__":
    main()
