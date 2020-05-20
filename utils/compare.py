from __future__ import absolute_import
import numpy as np
import cv2
from urllib.request import urlopen, Request
from .reframe import zoomclass
import matplotlib.pyplot as plt

from .predict import predict_label, LABEL
from .model import net_model
import os


def vector_feature(img, basemodel):
    """Convert the array img in a vector that can be analysed"""
    img1 = cv2.resize(img, (224, 224))
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
    return feature_vector


class get_pict:
    """Find the closest picture of img in the library of url lib"""


    def __init__(self, img, lib):
        """img : cv2 image
           lib : numpy array of url
           config : path to yolo config.cfg
           weights : path to yolo .weights
           label : path to yolo .names"""
        pict = cv2.imread(img)

        #the picture converted in a cv2 tab, its a list of tab, 
        #each tab is a zoom on a t-shirt, if there is no zoom there is no comparison
        self.imgs = zoomclass('t_shirt', pict)
        self.n = len(self.imgs)
        self.lib = lib

        self.img_close = {}
        self.img_dist = {}
        for i in range(self.n):
            self.img_close[i] = []
            self.img_dist[i] = []

        #net_model is the network that classifies between a t-shirt and a polo
        net_model.load_weights(os.getcwd() + '/utils/weights/weights')
        self.net_model = net_model

        #LABEL is a list of labels ['polo', 't-shirt']
        self.label_name = LABEL
        #we need the labels of the image to analyse
        self.label = [predict_label(img, net_model, LABEL) for img in self.imgs]


    def open_url(self, url):
        """
        Open url and load it in a cv2 image

        url : string
        """
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}
        req = Request(url=url, headers=headers)
        req = urlopen(req).read()
        img = np.asarray(bytearray(req), dtype="uint8")
        return cv2.imdecode(img, cv2.IMREAD_COLOR)

 
    def vect_compare(self, pict_lib, vect_imgs, basemodel, distance, threshold, curs, i):
        """
        Compare pict_lib to vect_imgs according to basemodel and distance
        """
        #normalize the picture
        blob = cv2.dnn.blobFromImage(pict_lib, 1/255.0, (416, 416), swapRB=True, crop=False)
        resize = blob[0].reshape((416, 416, 3))
        vect_pict_lib = vector_feature(resize, basemodel)
        compt = 0
        lb = predict_label(pict_lib, net_model, self.label_name)
        for j, vect_img in enumerate(vect_imgs):
            if lb == self.label[j]:
                dist_feat = distance(vect_img, vect_pict_lib)
                print("dist", dist_feat, " nb ", curs, end="\r")
                if dist_feat > threshold and i not in self.img_close:
                    self.img_close[compt].append(i)
                    self.img_dist[compt].append(dist_feat)
            compt += 1


    def compare(self, Model, network, layer, distance, threshold=0.7, yolo=False, treat=None):
        """
        Find the closest pictures of self.imgs in the library self.lib
        according to the network 'netword' and the distance 'distance'

        Model : the base model for prediction
        network : a CNN network
        layer : the layer of the network that is interesting
        distance : the distance to compare 2 picture
        threshold : (opt) float
        yolo : (opt) a boolean, if true each image of the dataset are recentred with yolo, it may be longer
        """
        basemodel = Model(inputs=network.input, outputs=layer)
        vect_imgs = []
        for i in self.imgs:
            #each tab of self.imgs is converted into a vector
            blob = cv2.dnn.blobFromImage(i, 1/255.0, (416, 416), swapRB=True, crop=False)
            resize = blob[0].reshape((416, 416, 3))
            print(resize.shape)
            vect_imgs.append(vector_feature(resize, basemodel))
        curs = 0
        for i in self.lib:
            curs += 1
            error = False
            try:
                pict_libs = self.open_url(i)
            except:
                error = True

            if not(error):
                if yolo:
                    pict_libs = zoomclass('t_shirt', pict_libs)
                    for pict_lib in pict_libs:
                        self.vect_compare(pict_lib, vect_imgs, basemodel, distance, threshold, curs, i)
                else:
                    self.vect_compare(pict_libs, vect_imgs, basemodel, distance, threshold, curs, i)

    def display(self):
        """Display the closest picture from self.imgs"""
        for i in range(self.n):
            k = len(self.img_close[i])
            plt.figure(figsize=(15, 80))
            plt.subplot(k+1, 3, 1)
            plt.imshow(self.imgs[i])
            compt = 2
            for j in range(k):
                plt.subplot(k+1, 3, compt)
                pict = self.open_url(self.img_close[i][j])
                plt.imshow(pict)
                plt.title(str(self.img_dist[i][j]))
                compt += 1
        plt.show()
