import sys
import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from PIL import Image
from lib.mepi import MEPI
from lib.mosaic import Mosaic
from lib.MGBI5 import MGBI_5
from lib.mgepi import MGEPI

CROP = True
EDGE = 10
SAVE = True
SHOW = True
SHOWG = False
POST = "_2x"
DELT = False

def read_img(img):
	return tf.convert_to_tensor(img, dtype=np.uint8)

@tf.function
def do_psnr(tf_img1, tf_img2):
	return tf.image.psnr(tf_img1, tf_img2, max_val=255)

def run(path):
    global CROP, EDGE, SAVE, SHOW, SHOWG, POST
    
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

    if DELT == True:
        epi = MEPI(mimage, 2)
        out = epi.Delta()
        SHOW = True
        SHOWG = True
        SAVE = False
    else:
        epi = MEPI(mimage, 2)
        out = epi.Algorithm()
    
    if CROP == False:
        #補原圖
        out[:EDGE, :, :] = oimage[:EDGE, :, :]
        out[:, :EDGE, :] = oimage[:, :EDGE, :]
        out[-1*EDGE:, :, :] = oimage[-1*EDGE:, :, :]
        out[:, -1*EDGE:, :] = oimage[:, -1*EDGE:, :]
    else:
        #裁減
        out = out[EDGE:-1*EDGE, EDGE:-1*EDGE, :]
        oimage = oimage[EDGE:-1*EDGE, EDGE:-1*EDGE, :]

    if SHOW == True:
        if SHOWG == True:
            if DELT == True:
                plt.imshow(out[...,0], cmap='hsv', vmin=-255, vmax=255)
            else:
                plt.imshow(out[...,1], cmap='hsv', vmin=0, vmax=255)
            plt.show()
            assert False
        else:
            plt.imshow(out)
        plt.show()

    filename, ext = os.path.splitext(path)
    if SAVE == True:
        im = Image.fromarray(out)
        im.save("result/" + os.path.basename(filename) + POST + ext)

    p = float(do_psnr(read_img(oimage), read_img(out)))
    s = os.path.basename(filename) + ": " + str(p)
    print(s)
    return [p, (s + "\n")]

def main():
    global SHOW, DELT
    if len(sys.argv) != 3:
        print("No input file!")
        return -1
    arg = sys.argv[1]

    if os.path.isfile(arg):
        files = [arg]
    elif os.path.isdir(arg):
        files = [os.path.join(arg, f) for f in os.listdir(arg)]
    else:
        print("File '" + arg + "' does not exist!")
        return -1
    
    files.sort()
    print("Files to be run:")
    print(*files, sep="\n")
    print("============================")
    
    num_cores = mp.cpu_count()
    if len(files) > 1:
        SHOW = False
        DELT = False
    
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
