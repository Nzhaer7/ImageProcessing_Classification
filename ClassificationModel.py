import tensorflow as tf

class TrainingModel():

    def Sequential(self,x,y,actFunc,fLayActFunc,catNum,optim,lss,mtrc,epch):
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
        self.model.compile(optimizer=optim, loss=lss, metrics=mtrc)

        self.model.fit(self.x_train, self.y_train, epochs=epch)

        self.val_loss, self.val_acc=self.model.evaluate(self.x_test,self.y_test)
        print(self.val_loss,self.val_acc)

        self.model.save("Trained_Sequential_Model.model")



