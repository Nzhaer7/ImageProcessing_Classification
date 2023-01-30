import PrepDataset as prd
import ClassificationModel as cm
import pickle as pkl
import tensorflow as tf

def imageModel():
    path=""
    category=[""]
    img_size=50
    dataType="image"
    clasNames=[""]

    training_data=prd.Preparation.prepData(dataType,path,category,img_size, clasNames)
    prd.Preparation.createTraniningData(training_data)

    X=pkl.open("image_X.pickle", "rb")
    Y=pkl.open("image_Y.pickle", "rb")

    actFunc=tf.nn.relu
    fLayActFunc=tf.nn.softmax
    catNum=2
    optim="adam"
    lssFunc="sparse_categorical_crossentrophy"
    mtrc=["accuracy"]
    epch=10

    cm.TrainingModel.Sequential(X, Y, actFunc, fLayActFunc, catNum, optim, lssFunc, mtrc, epch)

def textModel():
    path = ""
    category = [""]
    img_size = 50
    dataType = "text"
    clasNames = [""]

    training_data=prd.Preparation.prepData(dataType, path, category, img_size, clasNames)
    prd.Preparation.createTraniningData(training_data, dataType)

    X = pkl.open("text_X.pickle", "rb")
    Y = pkl.open("text_Y.pickle", "rb")

    testSize=0.2

    cm.TrainingModel.Naive_Bayes(X, Y, testSize)

