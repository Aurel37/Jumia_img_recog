from __future__ import absolute_import
import numpy as np
import cv2
import os
from .treat import centering


def zoomclass(classname='t_shirts', img, config="yolo-coco/yolov3-spp.cfg", weights="yolo-coco/t_shirt.weights", label="yolo-coco/t_shirt.names", threshold=0.3, cfd=0.3):
    """
    Return a list of tabs, each tab represents a zoom on a t-shirt presents in img
    classname : a string
    img : the picture to treat
    """
    LABEL = open(label).read().strip().split("\n")

    network = cv2.dnn.readNetFromDarknet(config, weights)

    ln = network.getLayerNames()
    ln = [ln[i[0] - 1] for i in network.getUnconnectedOutLayers()]

    (H, W) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    network.setInput(blob)
    layer_res = network.forward(ln)
    boxes = []
    confidences = []
    classIDs = []

    for output in layer_res:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)

            if LABEL[classID] == classname:
                confidence = scores[classID]
                if confidence > threshold:
                    box = detection[0:4]*np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, cfd, threshold)
    list_img = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            if w > W:
                w = W
            if h > H:
                h = H
            #finally the t_shirt is displayed
            res = centering(img, x, y, w, h)

            #res is a BGR picture, we change it to RGB
            #for i in range(n):
            #    for j in range(m):
            #        RED = res[i][j][2]
            #        res[i][j][2] = res[i][j][0]
            #        res[i][j][0] = RED
            list_img.append(res)
    return list_img
