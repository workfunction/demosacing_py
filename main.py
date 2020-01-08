from epi import EPI
from mosaic import Mosaic
from MGBI5 import MGBI_5
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math
import tensorflow as tf

if __name__ == "__main__":

    oim = Image.open("picture_set/kodim01.png")
    #im = oim.resize((int(oim.width/2), int(oim.height/2)), Image.BICUBIC)

    image = np.array(oim)

    #epi = EPI(image, im.width, im.height, 2)

    #out = epi.Algorithm()

    mos = Mosaic(image, oim.width, oim.height)
    out = mos.Algorithm()

    #mos = MGBI_5(out, oim.width, oim.height)
    #out = mos.Algorithm()

    plt.imshow(out, cmap='gray', vmin=0, vmax=255)
    plt.show()

    #im.save("result/kodim01.png")

