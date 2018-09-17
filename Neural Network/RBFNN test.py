## this code is refered from https://github.com/PetraVidnerova/rbf_keras

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras import optimizers
from rbflayer import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt
from ML_functions import extract_x_y, split_data

def load_data():    
    file = 'Referenced Ultimate Oranges Matrix.xlsx'
    x_data, y_data = extract_x_y(file)
    return x_data, y_data

if __name__ == "__main__":

    X, y = load_data() 
    x_train, y_train, x_test, y_test = split_data(X, y)
    model = Sequential()
    rbflayer = RBFLayer(10,
                        initializer=InitCentersRandom(X), 
                        betas=2.0,
                        input_shape=(228,))
    model.add(rbflayer)
    model.add(Dense(1))

    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam())

    model.fit(x_train, y_train,
              batch_size=50,
              epochs=100,
              verbose=1)

    y_pred = model.predict(x_test)

    print(rbflayer.get_weights())

    
    plt.scatter(y_test, y_pred)
#    plt.plot([-1,1], [0,0], color='black')
#    plt.xlim([-1,1])
    
#    centers = rbflayer.get_weights()[0]
#    widths = rbflayer.get_weights()[1]
#    plt.scatter(centers, np.zeros(len(centers)), s=20*widths)
    
    plt.show()
