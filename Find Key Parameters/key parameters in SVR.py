from ML_functions import extract_x_y, split_data
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.svm import SVR


file = 'Referenced Ultimate Oranges Matrix.xlsx'
x_data, y_data = extract_x_y(file)
x_train, y_train, x_test, y_test = split_data(x_data, y_data)
x_data = pd.DataFrame(x_data)

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
#    print(mean)
    return mean




def svr_CV_poly(dataX, dataY, kernel = 'poly', degree = 3, gamma = 0.00001, coef0 = 0.0, c=1000, epsilon=0.002, n_split = 10, plot = 0):
    
    kf = KFold(n_splits=10, random_state = None, shuffle = True)
    index = kf.split(dataX)
    r2 = []
    for train_index, test_index in index:
        x_train, x_test = dataX.loc[train_index], dataX.loc[test_index]
        y_train, y_test = dataY[train_index], dataY[test_index]
        clf = SVR(kernel = kernel, C=c, epsilon=epsilon, degree = degree,
                  gamma = gamma, coef0 = coef0)
        clf.fit(x_train, y_train.ravel())
        y_pre = clf.predict(x_test)
        if plot == 1:
            plt.scatter(y_test, y_pre)
            plt.show()
        r2.append(r2_score(y_test, y_pre))
    mean = np.mean(r2)
    print(mean)
    return mean

def svr_CV_sigmoid(dataX, dataY, kernel = 'sigmoid', gamma = 0.00001, coef0 = 0.0, c=1000, epsilon=0.002, n_split = 10, plot = 0):
    
    kf = KFold(n_splits=10, random_state = None, shuffle = True)
    index = kf.split(dataX)
    r2 = []
    for train_index, test_index in index:
        x_train, x_test = dataX.loc[train_index], dataX.loc[test_index]
        y_train, y_test = dataY[train_index], dataY[test_index]
        clf = SVR(kernel = kernel, C=c, epsilon=epsilon, gamma = gamma, coef0 = coef0)
        clf.fit(x_train, y_train.ravel())
        y_pre = clf.predict(x_test)
        if plot == 1:
            plt.scatter(y_test, y_pre)
            plt.show()
        r2.append(r2_score(y_test, y_pre))
    mean = np.mean(r2)
#    print(mean)
    return mean

def test_rbf():
    print("Test1: the parameters in svr function with kernel value 'rbf' ")
    print("Test1: testing the model variation:")
    rbf_accuracy = []
    for _ in range(100):
        rbf_accuracy.append(svr_CV(x_data, y_data))
    plt.plot(rbf_accuracy)
    plt.show()
    print("Test1: model variation: ",  np.var(rbf_accuracy))
        
    
    print("Test1: testing the parameter gamma in the range np.arange(1e-07, 1e-04, 1e-07)")
    rbf_gamma_accuracy = []
    for i in np.arange(1e-07, 1e-04, 1e-07):
        rbf_gamma_accuracy.append(svr_CV(x_data, y_data, gamma = i))
    plt.plot(rbf_gamma_accuracy)
    plt.show()
    print("Test1: gamma: ",  np.var(rbf_gamma_accuracy))
    
    
    print("Test1: testing the parameter c in the range np.arange(1, 10000, 100)")
    rbf_c_accuracy = []
    for i in np.arange(1, 10000, 100):
        rbf_c_accuracy.append(svr_CV(x_data, y_data, c = i))
    plt.plot(rbf_c_accuracy)
    plt.show()
    print("Test1: c: ",  np.var(rbf_c_accuracy))
    
    
    print("Test1: testing the parameter epsilon in the range np.arange(1e-07, 10, 0.1)")
    rbf_epsilon_accuracy = []
    for i in np.arange(1e-07, 10, 0.1):
        rbf_epsilon_accuracy.append(svr_CV(x_data, y_data, epsilon = i))
    plt.plot(rbf_epsilon_accuracy)
    plt.show()
    print("Test1: epsilon: ",  np.var(rbf_epsilon_accuracy))
    
    
    
    
    
    
def test_poly():
    print("Test3: the parameters in svr function with kernel value 'poly' ")
    print("Test3: testing the model variation:")
    rbf_accuracy = []
    for _ in range(100):
        rbf_accuracy.append(svr_CV_poly(x_data, y_data))
    plt.plot(rbf_accuracy)
    plt.show()
    print("Test3: model variation: ",  np.var(rbf_accuracy))
    
    
    print("Test3: testing the parameter gamma in the range np.arange(1e-07, 1e-03, 1e-05)")
    rbf_gamma_accuracy = []
    for i in np.arange(1e-07, 1e-04, 1e-06):
        rbf_gamma_accuracy.append(svr_CV_poly(x_data, y_data, gamma = i))
    plt.plot(rbf_gamma_accuracy)
    plt.show()
    print("Test3: gamma: ",  np.var(rbf_gamma_accuracy))
    
    
    print("Test3: testing the parameter degree in the range np.arange(1, 20, 1)")
    rbf_c_accuracy = []
    for i in np.arange(1, 20, 1):
        rbf_c_accuracy.append(svr_CV_poly(x_data, y_data, degree = i))
    plt.plot(rbf_c_accuracy)
    plt.show()
    print("Test3: degree: ",  np.var(rbf_c_accuracy))
    
    
    print("Test3: testing the parameter coef0 in the range np.arange(0, 10, 0.01)")
    rbf_c_accuracy = []
    for i in np.arange(0, 10, 0.01):
        rbf_c_accuracy.append(svr_CV_poly(x_data, y_data, coef0 = i))
    plt.plot(rbf_c_accuracy)
    plt.show()
    print("Test3: coef0: ",  np.var(rbf_c_accuracy))
    
        
    
    print("Test3: testing the parameter c in the range np.arange(1, 10000, 100)")
    rbf_c_accuracy = []
    for i in np.arange(1, 10000, 100):
        rbf_c_accuracy.append(svr_CV_poly(x_data, y_data, c = i))
    plt.plot(rbf_c_accuracy)
    plt.show()
    print("Test3: c: ",  np.var(rbf_c_accuracy))
    
    
    
    print("Test3: testing the parameter epsilon in the range np.arange(1e-07, 2, 0.02)")
    rbf_epsilon_accuracy = []
    for i in np.arange(1e-07, 10, 0.1):
        rbf_epsilon_accuracy.append(svr_CV_poly(x_data, y_data, epsilon = i))
    plt.plot(rbf_epsilon_accuracy)
    plt.show()
    print("Test3: epsilon: ",  np.var(rbf_epsilon_accuracy))


def test_sigmoid():
    print("Test4: the parameters in svr function with kernel value 'sigmoid' ")
    print("Test4: testing the model variation:")
    rbf_accuracy = []
    for _ in range(100):
        rbf_accuracy.append(svr_CV_sigmoid(x_data, y_data))
    plt.plot(rbf_accuracy)
    plt.show()
    print("Test4: model variation: ",  np.var(rbf_accuracy))
        
    
    print("Test4: testing the parameter gamma in the range np.arange(1e-07, 1e-04, 1e-07)")
    rbf_gamma_accuracy = []
    for i in np.arange(1e-07, 1e-04, 1e-07):
        rbf_gamma_accuracy.append(svr_CV_sigmoid(x_data, y_data, gamma = i))
    plt.plot(rbf_gamma_accuracy)
    plt.show()
    print("Test4: gamma: ",  np.var(rbf_gamma_accuracy))
    
    
    print("Test4: testing the parameter degree in the range np.arange(-50, 50, 1)")
    rbf_c_accuracy = []
    for i in np.arange(-50, 50, 1):
        rbf_c_accuracy.append(svr_CV_sigmoid(x_data, y_data, coef0 = i))
    plt.plot(rbf_c_accuracy)
    plt.show()
    print("Test4: degree: ",  np.var(rbf_c_accuracy))
    
    
    print("Test4: testing the parameter c in the range np.arange(1, 10000, 100)")
    rbf_c_accuracy = []
    for i in np.arange(1, 10000, 100):
        rbf_c_accuracy.append(svr_CV_sigmoid(x_data, y_data, c = i))
    plt.plot(rbf_c_accuracy)
    plt.show()
    print("Test4: c: ",  np.var(rbf_c_accuracy))
    
    
    print("Test4: testing the parameter epsilon in the range np.arange(1e-07, 10, 0.1)")
    rbf_epsilon_accuracy = []
    for i in np.arange(1e-07, 10, 0.1):
        rbf_epsilon_accuracy.append(svr_CV_sigmoid(x_data, y_data, epsilon = i))
    plt.plot(rbf_epsilon_accuracy)
    plt.show()
    print("Test4: epsilon: ",  np.var(rbf_epsilon_accuracy))
    
    

#run test functions    
#test_rbf()    
test_poly()
#test_sigmoid()  