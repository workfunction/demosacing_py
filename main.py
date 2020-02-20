import sys
import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from lib.mepi import MEPI
from lib.mepi_delta import MEPR
from lib.mosaic import Mosaic
from lib.MGBI5 import MGBI_5
from lib.mgepi import MGEPI

crop = True
edge = 10
save = True
show = False
showG = False
postfix = "_2x"

def read_img(img):
	return tf.convert_to_tensor(img, dtype=np.uint8)

@tf.function
def do_psnr(tf_img1, tf_img2):
	return tf.image.psnr(tf_img1, tf_img2, max_val=255)

def main():
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
    
    a = []
    f = open("result/demofile2.txt", "w")
    for path in files:
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

        epi = MEPI(mimage, 2)
        out = epi.Algorithm()
        
        if crop == False:
            #補原圖
            out[:edge, :, :] = oimage[:edge, :, :]
            out[:, :edge, :] = oimage[:, :edge, :]
            out[-1*edge:, :, :] = oimage[-1*edge:, :, :]
            out[:, -1*edge:, :] = oimage[:, -1*edge:, :]
        else:
            #裁減
            out = out[edge:-1*edge, edge:-1*edge, :]
            oimage = oimage[edge:-1*edge, edge:-1*edge, :]

        if show == True:
            if showG == True:
                plt.imshow(out[...,1], cmap='gray', vmin=0, vmax=255)
            else:
                plt.imshow(out)
            plt.show()

        filename, ext = os.path.splitext(path)
        if save == True:
            im = Image.fromarray(out)
            im.save("result/" + os.path.basename(filename) + postfix + ext)

        p = float(do_psnr(read_img(oimage), read_img(out)))
        a.append(p)
        s = os.path.basename(filename) + ": " + str(p)
        print(s)
        f.write(s + "\n")
        
    f.close()
    print("AVG = " + str(np.average(a)))

if __name__ == "__main__":
    main()
