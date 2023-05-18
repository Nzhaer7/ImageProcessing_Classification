from keras.optimizers import Adam
import İmageDataPrep as ImgPrep
import TextDataPrep as TxtPrep
import ClassificationModel as cm
import pickle as pkl
import tensorflow as tf

class MainExecution():
    def imageModel(self, path, category, img_size):
        self.Path = path
        self.Category = category
        self.Img_size = img_size

        self.Prep_data=ImgPrep.Preparation.prepData(self.Path,self.Category, self.Img_size)
        self.Training_data= ImgPrep.Preparation.createTraniningData(self.Prep_data, self.Img_size)

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

    def textClassificationModel(self, path, clssNames, dataPrpose):
        self.Path = path
        self.ClasNames = clssNames
        self.DataPrpose = dataPrpose

        self.Prep_data = TxtPrep.Preparation.prepData(self.Path, self.ClasNames, self.DataPrpose)
        self.Training_data=TxtPrep.Preparation.createTraniningData(self.Prep_data, self.DataPrpose, self.DataPrpose)

        self.X = pkl.load(open('textClftn_X.pickle', 'rb'))
        self.Y = pkl.load(open('textClftn_Y.pickle', 'rb'))

        self.testSize=0.2

        cm.TrainingModel.Naive_Bayes(cm.TrainingModel(), self.X, self.Y, self.testSize)

    def textLearningModel(self, path, dataPrpose, clssNames, VcbSize):
        self.Path = path
        self.ClasNames = clssNames
        self.DataPrpose = dataPrpose
        self.VcbSize= VcbSize

        self.Prep_data, self.VcbSize = TxtPrep.Preparation.prepData(self.Path, self.ClasNames, self.DataPrpose, self.VcbSize)
        self.Training_data=TxtPrep.Preparation.createTraniningData(self.Prep_data, self.DataPrpose, self.VcbSize)

        self.X=pkl.load(open('textLrnng_X.pickle', 'rb'))
        self.Y=pkl.load(open('textLrnng_Y.pickle', 'rb'))

        self.actFunc = tf.nn.relu
        self.fLayActFunc = tf.nn.softmax
        self.optim = Adam
        self.lss = "sparse_categorical_crossentrophy"
        self.epch = 150
        self.btch = 64

        cm.TrainingModel.LSTM_Sequential(cm.TrainingModel(), self.X, self.Y, self.actFunc, self.fLayActFunc, self.lss, self.optim, self.epch, self.btch, self.VcbSize)