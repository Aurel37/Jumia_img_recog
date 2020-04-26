from scipy.spatial import distance

import keras

from keras.models import Model


def cosine_simil(vector1, vector2):
    return 1 - distance.cosine(vector1, vector2)


vgg16 = keras.applications.vgg16.VGG16(include_top=True, weights="imagenet", input_shape=(224, 224, 3), pooling="max")

vgg19 = keras.applications.vgg19.VGG19(include_top=True, weights="imagenet", input_shape=(224, 224, 3), pooling="max")

