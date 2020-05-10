import numpy
import cv2

LABEL = ['polo', 'shirt', 't_shirt']


def predict_label(img, net_model, label):
    img1 = cv2.resize(img, (224, 224))
    predict = net_model.predict(img1.reshape(1, 224, 224, 3))
    maxi = predict[0][0]
    curs = 0
    test = 0
    for i, pred in enumerate(predict[0]):
        test += pred
        if pred > maxi:
            maxi = pred
            curs = i
    return label[curs]
