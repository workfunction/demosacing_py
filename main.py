import sys
import os
import math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from lib.mepi import MEPI
from lib.mepi_delta import MEPR
from lib.mosaic import Mosaic
from lib.MGBI5 import MGBI_5

def read_img(img):
	return tf.convert_to_tensor(img, dtype=np.uint8)

@tf.function
def do_psnr(tf_img1, tf_img2):
	return tf.image.psnr(tf_img1, tf_img2, max_val=255)

def main():
    if len(sys.argv) != 3:
        print("No input file!")
        return -1
    path = sys.argv[1]

    if not os.path.isfile(path):
        print("File '" + path + "' does not exist!")
        return -1
    
    oim = Image.open(path)
    oimage = np.array(oim, dtype=np.uint8)

    im = oim.resize((int(oim.width/2), int(oim.height/2)), Image.BICUBIC)
    image = np.array(im, dtype=np.uint8)

    mos = Mosaic(image, im.width, im.height)
    mimage = mos.Algorithm()

    epi = MEPI(mimage, 2)
    out = epi.Algorithm()

    #plt.imshow(out, cmap='gray', vmin=0, vmax=255)

    #mos = MGBI_5(out)
    #out = mos.Algorithm()

    #out[:3, :, :] = oimage[:3, :, :]
    #out[:, :3, :] = oimage[:, :3, :]
    #out[-3:, :, :] = oimage[-3:, :, :]
    #out[:, -3:, :] = oimage[:, -3:, :]

    #plt.imshow(out[...,1], cmap='gray', vmin=0, vmax=255)
    plt.imshow(out)
    plt.show()

    im = Image.fromarray(out)
    filename, ext = os.path.splitext(path)
    im.save("result/" + os.path.basename(filename) + "_2x" + ext)

    p = do_psnr(read_img(oimage), read_img(out))
    print(p)
    #im.save("result/kodim01.png")
    f = open("result/demofile2.txt", "a")
    f.write(os.path.basename(filename) + ": " + str(p) + "\n")
    f.close()

if __name__ == "__main__":
    main()
