from epi import EPI
from mosaic import Mosaic
from MGBI5 import MGBI_5
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math
import tensorflow as tf


def read_img(img):
	return tf.convert_to_tensor(img, dtype=np.uint8)
 
def do_psnr(tf_img1, tf_img2):
	return tf.image.psnr(tf_img1, tf_img2, max_val=255)
 
def psnr(t1, t2):
	with tf.compat.v1.Session() as sess:
		sess.run(tf.compat.v1.global_variables_initializer())
		return sess.run(do_psnr(t1, t2))

if __name__ == "__main__":
    oim = Image.open("picture_set/kodim01.png")
    #im = oim.resize((int(oim.width/2), int(oim.height/2)), Image.BICUBIC)

    image = np.array(oim, dtype=np.uint8)

    #epi = EPI(image, im.width, im.height, 2)

    #out = epi.Algorithm()

    mos = Mosaic(image, oim.width, oim.height)
    out = mos.Algorithm()

    #plt.imshow(out, cmap='gray', vmin=0, vmax=255)

    mos = MGBI_5(out)
    out = mos.Algorithm()

    out[:10, :, :] = image[:10, :, :]
    out[:, :10, :] = image[:, :10, :]
    out[-10:, :, :] = image[-10:, :, :]
    out[:, -10:, :] = image[:, -10:, :]

    plt.imshow(out[...,1], cmap='gray', vmin=0, vmax=255)
    plt.show()
    im = Image.fromarray(out)
    im.save("result/kodim01.png")

    p = psnr(read_img(image), read_img(out))
    print(p)
    #im.save("result/kodim01.png")

