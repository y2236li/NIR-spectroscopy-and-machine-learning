from sklearn.cross_decomposition import PLSRegression
from ML_functions import extract_x_y, split_data
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd



file = 'Referenced Ultimate Oranges Matrix.xlsx'
x_data, y_data = extract_x_y(file)
x_train, y_train, x_test, y_test = split_data(x_data, y_data)

def pls_CV(dataX, dataY, max_iter, n_components, tol, plot = 0):
    dataX = pd.DataFrame(dataX)
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
    print(mean)
    return mean



max_mean_set = [0]*20
## By changing the following code to indentify the impact of parameters
for _ in range(100):
    mean_set = []
    for i in np.arange(1, 20, 1):
        mean_set.append(pls_CV(x_data, y_data, max_iter = 500, n_components = i, tol = 1e-06,
                               plot=0))
    plt.plot(mean_set)
    plt.show()
    max_mean_set[np.argmax(mean_set)] += 1
print("Showing frequency of the n_components value achieved the highest accuracy")
plt.bar(np.arange(1,21,1), np.array(max_mean_set))
plt.show()