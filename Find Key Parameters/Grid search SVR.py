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

def svr_CV(dataX, dataY, gamma = 8.99e-05, c=1000, epsilon = 4.8e-06, n_split = 10, plot = 0):
    
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


def Grid_search_rbf(dataX, dataY, gamma_list, epsilon_list):
    print("Doing grid search on SVR_rbf")
    accuracy_list_tmp = []
    gamma_list_tmp = []
    epsilon_list_tmp = []
    for gamma in gamma_list:
        for epsilon in epsilon_list:
            accuracy = svr_CV(dataX, dataY, gamma = gamma, epsilon = epsilon)
            accuracy_list_tmp.append(accuracy)
            gamma_list_tmp.append(gamma)
            epsilon_list_tmp.append(epsilon)
    plt.plot(accuracy_list_tmp)
    plt.show()
    max_index = np.argmax(accuracy_list_tmp)
    print("Highest accuracy: ", accuracy_list_tmp[max_index])
    print("Default value of gamma and epsilon are ", gamma_list_tmp[max_index],
          " and ", epsilon_list_tmp[max_index])
    
    
def Grid_search_poly(dataX, dataY, gamma_list, epsilon_list, degree_list, coef0_list, c_list):
    print("Doing grid search on SVR_rbf")
    accuracy_list_tmp = []
    gamma_list_tmp = []
    epsilon_list_tmp = []
    degree_list_tmp = []
    coef0_list_tmp = []
    c_list_tmp = []
    for gamma in gamma_list:
        for epsilon in epsilon_list:
            for degree in degree_list:
                for coef0 in coef0_list:
                    for c in c_list:
                        accuracy = svr_CV_poly(dataX, dataY, gamma = gamma, epsilon = epsilon,
                                          degree = degree, coef0 = coef0, c = c)
                        accuracy_list_tmp.append(accuracy)
                        gamma_list_tmp.append(gamma)
                        epsilon_list_tmp.append(epsilon)
                        degree_list_tmp.append(degree)
                        coef0_list_tmp.append(coef0)
                        c_list_tmp.append(c)
    plt.plot(accuracy_list_tmp)
    plt.show()
    max_index = np.argmax(accuracy_list_tmp)
    print("Highest accuracy: ", accuracy_list_tmp[max_index])
    print("Default value of gamma, epsilon, degree, coef0, c", gamma_list_tmp[max_index],
          epsilon_list_tmp[max_index], degree_list_tmp[max_index], coef0_list_tmp[max_index],
          c_list_tmp[max_index])
            
def default_value_rbf():
    gamma_list = np.arange(1e-07, 1e-04, 1e-07)
    epsilon_list = np.arange(1e-07, 1.5, 1e-02)
    Grid_search_rbf(x_data, y_data, gamma_list, epsilon_list)
    
def default_value_poly():
    gamma_list = np.arange(1e-07, 1e-04, 1e-06)
    epsilon_list = np.arange(1e-07, 1.5, 1e-02)
    c_list = np.arange(5000, 10000, 100)
    degree_list = np.arange(1, 5, 1)
    coef0_list = np.arange(1e-07, 2, 0.1)
    Grid_search_poly(x_data, y_data, gamma_list, epsilon_list, degree_list, coef0_list, c_list)
            
accuracy_list = []
gamma_list = np.arange(1e-07, 1e-04, 1e-06)
epsilon_list = np.arange(1e-07, 1e-5, 1e-07)
degree_list = np.arange(1, 5, 1)
coef0_list = np.arange(1e-07, 2, 0.1)
c_list = np.arange(5000, 10000, 100)
for i in c_list:
    accuracy_list.append(svr_CV_poly(x_data, y_data, gamma = 3.21e-05, epsilon = 5.2e-06, 
                                     degree = 2, coef0 = 1.6000001, c = i))
plt.plot(accuracy_list)
plt.show()
max_index = np.argmax(accuracy_list)
print(c_list[max_index])
            