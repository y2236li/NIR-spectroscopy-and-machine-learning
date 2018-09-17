from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from ML_functions import extract_x_y, split_data


def plot_loss(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + 
                     str(str(format(history.history[l][-1],'.5f'))+')'))


def train_model_FFNN(file):
    file = file
    x_data, y_data = extract_x_y(file)
    
    
    
    x_train, y_train, x_test, y_test = split_data(x_data, y_data)
    
    
    model = Sequential()
    model.add(Dense(units=110, activation='relu',
                    input_dim=len(x_train[0]), kernel_initializer='random_uniform'))
    model.add(LeakyReLU(alpha=.002))
#    model.add(Dense(units=180, activation='relu',
#                    input_dim=len(x_train[0]), kernel_initializer='random_uniform'))
#    model.add(LeakyReLU(alpha=.002))
    model.add(Dense(units=1, activation='relu', kernel_initializer='random_uniform'))
    
    model.compile(loss='squared_hinge',
                  #optimizer='SGD'
                  #optimizer='RMSprop' #a good choice for recurrent neural networks.
                  #optimizer='Adagrad' #The more updates a parameter receives, the smaller the updates.
                  #optimizer='Adadelta' #Adadelta continues learning even when many updates have been done
                  #optimizer='adam'
                  #optimizer='Adamax' #It is a variant of Adam based on the infinity norm.
                  optimizer='Nadam' #Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop with Nesterov momentum.
                  #keras.optimizers.TFOptimizer(optimizer) Wrapper class for native TensorFlow optimizers.
                  )
    
    
    history = model.fit(x_train, y_train, epochs=100, batch_size=int(len(x_train)/10), verbose = 1)
    
    plot_loss(history)
    y_pre = model.predict(x_test)
    
    plt.show()
    plt.scatter(y_test, y_pre)
    plt.show()
    print("R^2: ", r2_score(y_test, y_pre))
    
file = 'Referenced Ultimate Oranges Matrix.xlsx'
train_model_FFNN(file)