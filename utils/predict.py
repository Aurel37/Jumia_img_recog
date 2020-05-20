import numpy
import cv2

LABEL = ['polo', 't_shirt']

def predict_label(img, net_model, label):
    """Return the label of the picture img"""
    img1 = cv2.resize(img, (80, 80))
    predict = net_model.predict(img1.reshape(1, 80, 80, 3))
    maxi = predict[0][0]
    curs = 0
    test = 0
    for i, pred in enumerate(predict[0]):
        test += pred
        if pred > maxi:
            maxi = pred
            curs = i
    return label[curs]
