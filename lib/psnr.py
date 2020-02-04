import tensorflow as tf
import numpy as np

def read_img(path):
	return tf.image.decode_png(tf.io.read_file(path), channels=3)
 
def do_psnr(tf_img1, tf_img2):
	return tf.image.psnr(tf_img1, tf_img2, max_val=255)
 
if __name__ == "__main__":
	with tf.compat.v1.Session() as sess:
		t1 = read_img('t1.png')
		t2 = read_img('t2.png')
		sess.run(tf.compat.v1.global_variables_initializer())
		p = sess.run(do_psnr(t1, t2))
		print(p)