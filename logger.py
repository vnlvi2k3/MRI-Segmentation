import tensorflow as tf
import numpy as np
from io import BytesIO
import imageio

class Logger(object):

    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()

    def image_summary(self, tag, image, step):
        s = BytesIO()
        imageio.imwrite(s, image, format="png")

        with self.writer.as_default():
            tf.summary.image(name=tag, data=np.array([image]), step=step)
            self.writer.flush()

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return
        with self.writer.as_default():
            for i, img in enumerate(images):
                tf.summary.image(name=f"{tag}/{i}", data=np.array([img]), step=step)
            self.writer.flush()