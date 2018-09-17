from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from Input_process import Data_operation

######################################################################################################
#this is a library used to save the dicriminant methods
######################################################################################################

class Discriminant_functions():
    
    
    def svr_CV(self, dataX, dataY, gamma = 8.99e-05, c=1000, epsilon = 4.8e-06, n_split = 10, plot = 0):
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
#        print("   ", mean)
        return mean

    def svr_CV_poly(self, dataX, dataY, kernel = 'poly', c = 6100, gamma = 3.21e-05,
                    coef0 = 1.6000001, degree = 2, epsilon = 5.31e-05, n_split = 10, plot = 0):
        
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
#        print("   ", mean)
        return mean
    
    
    def decision_tree (self, x_data, y_data, criterion_index = 0, max_depth = None,
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
    
    
    def decision_tree_CV(self, dataX, dataY, criterion_index = 0, max_depth = None,
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
            clf = self.decision_tree(x_train, y_train, criterion_index = criterion_index,
                                max_depth = max_depth, min_samples_split = min_samples_split,
                                min_samples_leaf = min_samples_leaf, max_features = max_features)
            clf.fit(x_train, y_train)
            y_pre = clf.predict(x_test)
            scores = accuracy_score(y_test, y_pre.round())
            r2.append(scores)
        mean = np.mean(r2)
#        print("   ", mean)
        return mean
    
    
    def pls_CV(self, dataX, dataY, n_components, max_iter = 500,  tol = 1e-06, plot = 0):
#        dataX = pd.DataFrame(dataX)
        if (n_components > len(dataX.columns)): n_components = len(dataX.columns)
        kf = KFold(n_splits=10, random_state = None, shuffle = True)
        index = kf.split(dataX)
        r2 = []
        for train_index, test_index in index:
            x_train, x_test = dataX.loc[train_index], dataX.loc[test_index]
            y_train, y_test = dataY[train_index], dataY[test_index]
            pls2 = PLSRegression(copy=True, max_iter=max_iter, n_components=n_components, scale=True,
            tol=tol)
            pls2.fit(x_train, y_train)
            
            y_pred = pls2.predict(x_test)
            if plot == 1:
                plt.scatter(y_test, y_pred)
                plt.show()
            r2.append(r2_score(y_test, y_pred))
        mean = np.mean(r2)
#        print("   ", mean)
        return mean
    
    
    
    
    
    
    
