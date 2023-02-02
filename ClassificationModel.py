import tensorflow as tf
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle


class TrainingModel():

    def Sequential(self,x,y,actFunc,fLayActFunc,catNum,optim,lssFunc,mtrc,epch):
        self.x_train=x
        self.y_train=y
        self.x_test=x
        self.y_test=y

        self.x_train=tf.keras.utils.normalize(self.x_train, axis=1)
        self.x_test=tf.keras.utils.normalize(self.x_test,axis=1)

        self.model=tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation=actFunc))
        self.model.add(tf.keras.layers.Dense(128, activation=actFunc))
        self.model.add(tf.keras.layers.Dense(128, activation=actFunc))
        self.model.add(tf.keras.layers.Dense(catNum, activation=fLayActFunc))

        self.model.compile(optimizer=optim, loss=lssFunc, metrics=mtrc)

        self.model.fit(self.x_train, self.y_train, epochs=epch)

        self.val_loss, self.val_acc=self.model.evaluate(self.x_test,self.y_test)
        print(self.val_loss,self.val_acc)

        self.model.save("Trained_Sequential_Model.model")

    def LSTM_Sequential(self,x,y,actFunc,fLayActFunc,lss,optim,epch,btch,vcbSize):
        self.x=x
        self.y=y

        self.model=tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Embedding(vcbSize, 10, input_length=1))
        self.model.add(tf.keras.layers.LSTM(1000, return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(1000))
        self.model.add(tf.keras.layers.Dense(1000, activation=actFunc))
        self.model.add(tf.keras.layers.Dense(vcbSize, activation=fLayActFunc))

        self.model.compile(loss=lss, optimizer=optim(lr="0.001"))

        self.model.fit(self.x, self.y, epochs=epch, batch_size=btch)

        self.model.save("Trained_Text_Learning_Model.model")


    def Naive_Bayes(self, x, y, tstSize):
        self.cv=CountVectorizer()
        self.X=self.cv.fit_transform(x)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X,y, test_size=tstSize)

        self.model=MultinomialNB()

        self.model.fit(self.x_train, self.y_train)

        self.result=self.model.score(self.x_test, self.y_test)

        print(self.result*100)

        pickle.dump(self.model, open("trained_Naive_Bayes_Model.pkl", "wb"))
        pickle.dump(self.cv, open("cVectorizer.pkl", "wb"))







