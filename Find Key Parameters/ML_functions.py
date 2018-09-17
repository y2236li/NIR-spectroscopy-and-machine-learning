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




def plot_loss(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + 
                     str(str(format(history.history[l][-1],'.5f'))+')'))

def split_data(x_data, y_data):
    train = np.random.randint(len(x_data), size = int(len(x_data)*0.7))
    x_train = []
    y_train = []
    x_test = []
    y_test = [] 
    for i in range(len(x_data)):
        if i in train:
            x_train.append(x_data[i])
            y_train.append(y_data[i])
        else:
            x_test.append(x_data[i])
            y_test.append(y_data[i])
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def extract_x_y(file):
    xl = pd.ExcelFile(file)
    df = xl.parse('Sheet1') #create a dataframe
    brix = df[df.columns[0]]
    df = df.drop(df.columns[0], axis = 1, inplace = False)
    x_data = []
    y_data = []
    for i in range(df.shape[0]):
        new_xdata = list(df.loc[i])
        x_data.append(new_xdata)
        new_ydata = [brix[i]]
        y_data.append(new_ydata)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data
    


def train_model_ANN(file, units):
    file = file
    x_data, y_data = extract_x_y(file)
    
    
    
    x_train, y_train, x_test, y_test = split_data(x_data, y_data)
    
    
    model = Sequential()
    model.add(Dense(units=110, activation='relu',
                    input_dim=len(x_train[0]), kernel_initializer='random_uniform'))
    model.add(LeakyReLU(alpha=.002))
    model.add(Dense(units=180, activation='relu',
                    input_dim=len(x_train[0]), kernel_initializer='random_uniform'))
    model.add(LeakyReLU(alpha=.002))
    model.add(Dense(units=1, activation='relu', kernel_initializer='random_uniform'))
    
    model.compile(loss='squared_hinge',
                  #optimizer='SGD'
                  #optimizer='RMSprop' #a good choice for recurrent neural networks.
                  #optimizer='Adagrad' #The more updates a parameter receives, the smaller the updates.
                  #optimizer='Adadelta' #Adadelta continues learning even when many updates have been done
                  optimizer='adam'
                  #optimizer='Adamax' #It is a variant of Adam based on the infinity norm.
                  #optimizer='Nadam' #Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop with Nesterov momentum.
                  #keras.optimizers.TFOptimizer(optimizer) Wrapper class for native TensorFlow optimizers.
                  )
    
    
    history = model.fit(x_train, y_train, epochs=100, batch_size=int(len(x_train)/10), verbose = 1)
    
    plot_loss(history)
    y_pre = model.predict(x_test)
    
    plt.show()
    plt.scatter(y_test, y_pre)
    plt.show()
    print("R^2: ", r2_score(y_test, y_pre))
    
    
    
def svr_CV(dataX, dataY, gamma=0.00001, c=1000, epsilon=0.002, n_split = 10, plot = 0):
    
    kf = KFold(n_splits=10, random_state = None, shuffle = True)
    index = kf.split(dataX)
    r2 = []
    for train_index, test_index in index:
        x_train, x_test = dataX.loc[train_index], dataX.loc[test_index]
        y_train, y_test = dataY[train_index], dataY[test_index]
        clf = SVR(C=c, epsilon=epsilon, gamma = gamma)
        clf.fit(x_train, y_train.ravel())
        y_pre = clf.predict(x_test)
        if plot == 1:
            plt.scatter(y_test, y_pre)
            plt.show()
        r2.append(r2_score(y_test, y_pre))
    mean = np.mean(r2)
    print(mean)
    return mean
    
    
def Grid_search(dataX, dataY, func, params):
    clf = GridSearchCV(func, params, scoring = 'r2', verbose = 100)
    return clf
    
    



#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
#print(loss_and_metrics)











