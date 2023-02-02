from keras.optimizers import Adam
import PrepDataset as prd
import ClassificationModel as cm
import pickle as pkl
import tensorflow as tf

class MainExecution():
    def imageModel(self, path, category, img_size, dataType, dataPrpose, clasNames):
        self.Path = path
        self.Category = category
        self.Img_size = img_size
        self.DataType = dataType
        self.ClasNames = clasNames
        self.DataPrpose = dataPrpose
        self.VcbSize=0

        self.training_data=prd.Preparation.prepData(prd.Preparation(), self.DataType, self.DataPrpose, self.Path, self.Category, self.Img_size, self.ClasNames)
        prd.Preparation.createTraniningData(prd.Preparation(), self.training_data, self.DataType, self.DataPrpose, self.Img_size,self.VcbSize)

        self.X = pkl.load(open('Image_X.pickle', 'rb'))
        self.Y = pkl.load(open('Image_Y.pickle', 'rb'))

        self.actFunc=tf.nn.relu
        self.fLayActFunc=tf.nn.softmax
        self.catNum=2
        self.optim="adam"
        self.lssFunc="sparse_categorical_crossentrophy"
        self.mtrc=["accuracy"]
        self.epch=10

        cm.TrainingModel.Sequential(cm.TrainingModel(), self.X, self.Y, self.actFunc, self.fLayActFunc, self.catNum, self.optim, self.lssFunc, self.mtrc, self.epch)

    def textClassificationModel(self, path, dataType, dataPrpose):
        self.Path = path
        self.Category = [""]
        self.Img_size = 0
        self.DataType = dataType
        self.ClasNames = [""]
        self.DataPrpose = dataPrpose
        self.VcbSize = 0

        self.training_data = prd.Preparation.prepData(prd.Preparation(), self.DataType, self.DataPrpose, self.Path, self.Category, self.Img_size, self.ClasNames)
        prd.Preparation.createTraniningData(prd.Preparation(), self.training_data, self.DataType, self.DataPrpose, self.Img_size,self.VcbSize)

        self.X = pkl.load(open('textClftn_X.pickle', 'rb'))
        self.Y = pkl.load(open('textClftn_Y.pickle', 'rb'))

        self.testSize=0.2

        cm.TrainingModel.Naive_Bayes(cm.TrainingModel(), self.X, self.Y, self.testSize)

    def textLearningModel(self, path, dataType, dataPrpose):
        self.Path = path
        self.Category = [""]
        self.Img_size = 0
        self.DataType = dataType
        self.ClasNames = [""]
        self.DataPrpose = dataPrpose

        self.training_data, self.VcbSize = prd.Preparation.prepData(prd.Preparation(), self.DataType, self.DataPrpose, self.Path, self.Category, self.Img_size, self.ClasNames)
        prd.Preparation.createTraniningData(prd.Preparation(), self.training_data, self.DataType, self.DataPrpose, self.Img_size,self.VcbSize)

        self.X=pkl.load(open('textLrnng_X.pickle', 'rb'))
        self.Y=pkl.load(open('textLrnng_Y.pickle', 'rb'))

        self.actFunc = tf.nn.relu
        self.fLayActFunc = tf.nn.softmax
        self.optim = Adam
        self.lss = "sparse_categorical_crossentrophy"
        self.epch = 150
        self.btch = 64

        cm.TrainingModel.LSTM_Sequential(cm.TrainingModel(), self.X, self.Y, self.actFunc, self.fLayActFunc, self.lss, self.optim, self.epch, self.btch, self.VcbSize)