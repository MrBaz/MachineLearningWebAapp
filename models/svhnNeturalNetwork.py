import numpy as np
import h5py
import pickle
# import tensorflow as tf
# import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Reshape, BatchNormalization
from tensorflow.keras import optimizers
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.preprocessing.image import  img_to_array, load_img

# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True
from PIL import Image
from numpy import asarray


class SVHNEstimator():
    """
       Comment out class functions and purpose, very useful to remember and to other readers.
    """
    def __init__(self):
        """
        Constructor code goes here  
        """
        
        # self. .... 

    def loadDataset(self):
        f = h5py.File('data/SVHN_single_grey1.h5','r')

        x_train = np.array(f['X_train'])
        y_train = np.array(f['y_train'])
        x_val = np.array(f['X_val'])
        y_val = np.array(f['y_val'])
        x_test = np.array(f['X_test'])
        y_test = np.array(f['y_test'])

        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))
        x_val = x_val.reshape((x_val.shape[0], -1))

        y_train = to_categorical(y_train)
        y_train = y_train.reshape((y_train.shape[0], -1))

        y_val = to_categorical(y_val)
        y_val = y_val.reshape((y_val.shape[0], -1))

        y_test = to_categorical(y_test)
        y_test = y_test.reshape((y_test.shape[0], -1))

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def build(self):
        model = Sequential()

        model.add(Dense(256, input_shape = (1024,), activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(128, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(64, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(32, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(10, activation = 'softmax'))

        opt = optimizers.Adam(lr = 0.01 , beta_1=0.9 , decay =0)
        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model


    def fit(self, no_of_epochs=2, pre_batch_size=200 ):
       
        model = self.model
        history = model.fit(self.x_train, self.y_train, batch_size = 1000, epochs=no_of_epochs, verbose = 1,validation_data=(self.x_val, self.y_val))
        # Save Model after training
        self.model = model
        
        Y_pred_cls = model.predict_classes(self.x_test, batch_size=pre_batch_size, verbose=0)

        print('Accuracy') 
        print( str(model.evaluate(self.x_test, self.y_test)[1]) )
        label=np.argmax(self.y_test.T, axis=0)

        print(confusion_matrix(label, Y_pred_cls))
        print(classification_report(Y_pred_cls, label))
        return history

    def saveTrainedModel(self):
        self.model.save('SVHN-NN.h5')
        # with open('SVHN-NN.dat', 'wb') as f:
        #     pickle.dump(self.model, f)

    def loadTrainedModel(self):
        self.model = load_model('SVHN-NN.h5')
        # with open('SVHN-NN.dat', 'rb') as f:
	    #     self.model = pickle.load(f)

    def predict(self, inputImage):
        img = load_img(inputImage)  # this is a PIL image
        # img.thumbnail((128, 128))
        # img.show()

        # Convert to Numpy Array
        x = img_to_array(img)  
       
        x = x.reshape((x.shape[2], -1))
        
        
        prediction = self.model.predict_classes(x, verbose=0)
        label= prediction[1]

        
        # prediction=None
        print('label:')
        print(label)
      
        return label

class SVMEstimator():
     def __init__(self):
        """
        Constructor code goes here  
        """