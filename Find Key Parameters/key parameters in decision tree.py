
# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from Input_process import Data_operation
from sklearn.model_selection import KFold





# the y data was preprocessed into one-hot binary numbers.
# the default length of the one-hot binary numbers is 10


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





def decision_tree (x_data, y_data, criterion_index = 0, max_depth = None,
                   min_samples_split = 2, min_samples_leaf = 1, max_features = None):

    if (criterion_index == 0):
        criterion = 'mse'
    elif (criterion_index == 1):
        criterion = 'friedman_mse'
    else:
        criterion = 'mae'
    
    # Fit regression model
    regr_2 = DecisionTreeRegressor(criterion = criterion,  max_depth=max_depth)
    
    return regr_2


def decision_tree_CV(dataX, dataY, criterion_index = 0, max_depth = None,
                     min_samples_split = 2, min_samples_leaf = 1,  n_split = 7, 
                     max_features = None):
    kf = KFold(n_splits=n_split, random_state = None, shuffle = True)
    index = kf.split(dataX)
    r2 = []
    data_operation = Data_operation(dataY)
    dataY_binary = pd.DataFrame(data_operation.to_binary())
    for train_index, test_index in index:
        x_train, x_test = dataX.loc[train_index], dataX.loc[test_index]
        y_train, y_test = dataY_binary.loc[train_index], dataY_binary.loc[test_index]
        clf = decision_tree(x_train, y_train, criterion_index = criterion_index,
                            max_depth = max_depth, min_samples_split = min_samples_split,
                            min_samples_leaf = min_samples_leaf, max_features = max_features)
        clf.fit(x_train, y_train)
        y_pre = clf.predict(x_test)
        scores = accuracy_score(y_test, y_pre.round())
        r2.append(scores)
    mean = np.mean(r2)
    print(mean)
    return mean
    




file = 'Referenced Ultimate Oranges Matrix.xlsx'
x_data, y_data = extract_x_y(file)
x_data = pd.DataFrame(x_data)

accuracy_list = []
for i in np.arange(1, 100, 1):
    accuracy_list.append(decision_tree_CV(x_data, y_data, criterion_index = 0,
                                          max_features = 'sqrt'))
plt.plot(accuracy_list)
plt.show()





