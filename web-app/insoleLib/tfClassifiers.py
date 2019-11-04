import os
import pandas as pd
import numpy as np
from .columns import DataColumns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import keras

import matplotlib.pyplot as plt
import seaborn as sn

# https://www.tensorflow.org/tutorials/keras/basic_classification

# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

class TfClassifiers:
    """
    Unfinished since ran out of time.
    Neaural network classifiers that are implemented with tensorflow/keras.
    """

    @staticmethod
    def testNn(data, x_cols, y_cols, plots=False, orig_acc=0, orig_auc=0):
        """
        Testing basic neural network classifier with two hidden layers
        
        Arguments:
            data {[type]} -- [description]
            x_cols {[type]} -- [description]
            y_cols {[type]} -- [description]
        
        Keyword Arguments:
            plots {bool} -- [description] (default: {False})
            orig_acc {double} -- [description] (default: {0})
            orig_auc {double} -- [description] (default: {0})
        """
        
        
        data = shuffle(data)
        x = data.loc[:, x_cols]
        y = data.loc[:, y_cols]
        
        y_dummies = pd.get_dummies(y) #to binary labels
        bin_cols = ["label_Normal","label_Fall"]
        
        dimensions = x.shape[1]
        print("dimensions ", dimensions)
        
        ##encode class values as integers
        #encoder = LabelEncoder()
        #encoder.fit(y)
        #y = encoder.transform(Y)
        #dummy_y = np_utils.to_categorical(y) # convert to dummy one hot encoding
        
        y = y_dummies
        
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0, shuffle=True) #used for suffling for now
        
        #Defining model
        model = Sequential()
        model.add(Dense(dimensions, input_dim=dimensions, activation='relu'))
        model.add(Dense(dimensions, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        
        #Compiling the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #early stop
        early_stop = EarlyStopping(monitor='val_loss', patience=80, verbose=1, mode="auto")
        
        accuracy_a = []
        real_label = []
        pred_label = []
        
        #Fitting model
        model.fit(xtrain, ytrain, epochs=200, batch_size=10, callbacks=[early_stop], validation_split = 0.3, verbose=1)
        
        np_xtest = np.array(xtest) #converting to numpy array
        ypred = model.predict(xtest, verbose=1)
        
        pred_label.append(ypred)
        real_label.append(ytest.values)
        
        ytest_nondum = ytest.idxmax(axis=1)
        print("ytest", ytest)
        print("ypred", ypred)
        
        acc = accuracy_score(ytest_nondum, ypred)
        accuracy_a.append(acc)
        
        avg_acc = np.mean(accuracy_a)
        print("NN Average accuracy ", avg_acc)
        
        pred_label_df = pd.DataFrame(columns=["label"])
        real_label_df = pd.DataFrame(columns=["label"])
        
        #Forming the dataframes
        #for row in range(0,len(pred_label)):
        #    label_str = pred_label[row][0]
        #    pred_label_df.loc[row] = label_str
        #
        #for row in range(0,len(real_label)):
        #    label_str = real_label[row][0][0]
        #    real_label_df.loc[row] = label_str
        
        #if(plots):
        #    #accuracy
        #    plt.plot(model.history['acc'])
        #    plt.plot(model.history['val_acc'])
        #    plt.title('Model accuracy')
        #    plt.ylabel('accuracy')
        #    plt.xlabel('epoch')
        #    plt.legend(['train', 'test'], loc='best')
        #    plt.show()
        #
        #    #loss
        #    plt.plot(model.history['loss'])
        #    plt.plot(model.history['val_loss'])
        #    plt.title('Model loss')
        #    plt.ylabel('loss')
        #    plt.xlabel('epoch')
        #    plt.legend(['train', 'test'], loc='best')
        #    plt.show()
        #    
        #    cm = confusion_matrix(real_label_df, pred_label_df)
        #    cm_df = pd.DataFrame(cm, ["Fall", "Normal"], ["Fall", "Normal"])
        #    
        #    sn.set(font_scale=1.5)
        #    sn.heatmap(cm_df, annot=True, annot_kws={"size": 32}, fmt='d')
        #    plt.savefig("../figs/tf_heatmap.png", facecolor="w", bbox_inches="tight")
        #    plt.show()

        #avg_acc = np.mean(accuracy_a)
        #
        ##Checking accuracy
        #print("Tree average accuracy: ", round(avg_acc, 2)) #2 decimals
        #
        ##More detailed report
        #print(classification_report(real_label_df, pred_label_df))

        return(avg_acc, real_label, pred_label)