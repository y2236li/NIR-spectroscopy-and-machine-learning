import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.fftpack import fft, dct
from sklearn.decomposition import PCA
from ML_functions import svr_CV, extract_x_y

import sys
import random
from sklearn.preprocessing import StandardScaler
from scipy import signal



class Preprocessing():
    def __init__(self, file):
        x_data, y_data = extract_x_y(file)
        x_data = pd.DataFrame(x_data)
        self.brix = y_data
        self.df = x_data
        self.file = file
        self.col = self.df.columns
        self.row = self.df.index
        
        
    def Moving_average_smoothing(self, windows_size = 3, center = 0):
        print("Moving_average_smoothing")
        centered = False
        if (windows_size >= len(self.df.columns)):
            print("Warning: number of columns is smaller than windows." +
                  "Skipped Moving average smoothing")
            return self.df
        for i in range(len(self.df.index)):
            if center != 0:
                centered = True
            else:
                centered = False
            self.df.loc[i] = pd.rolling_mean(self.df.loc[i], windows_size, center = centered)
        self.df = self.df.dropna(axis = 'columns')
        return self.df
    
        
        
    def MSC (self):
        print("MSC")
        df = self.df
        col = self.col
        row = self.row
        mean_row = []
        stdev_row = []
        b = []
        a = []
        for i in range (0, len(col)):
            mean = np.mean(self.df[col[i]])
            stdev = np.std (self.df[col[i]])
            mean_row.append(mean)
            stdev_row.append(stdev)
        df.append(mean_row)
        df.append(stdev_row)
        mean_row = df.loc[row[-2]]
        for i in range (0, len(row)):
            cur_row = df.loc[row[i]]
            mean_row = sm.add_constant(mean_row)
            regr = sm.OLS(cur_row, mean_row) #mean_row has to be the y value
            results = regr.fit()
            a.append(results.params[len(self.df.index)-2])
            b.append(results.params['const'])
        for i in range (0, len(col)):
            df[col[i]] = (df[col[i]]-b)/a #b: intercept, a:slope
        self.df = df
        return df
    
    
    def detrend(self):
        print("Detrend")
        self.df = signal.detrend(self.df)
        return self.df
    
    def SNV (self):
        print("SNV")
        df = self.df
        df = StandardScaler().fit_transform(df)
        self.df = pd.DataFrame(df)
        return df
    
    
    
    def DCT (self, typo=1):
        print("DCT")
        df = self.df
        for i in df.index:
            df.loc[i] = dct(np.array(df.loc[i]), typo)
        self.df = df
        return df
    
    def PCA(self, n_com = -1, plot = 0):
        print("PCA")
        df = self.df
        if (n_com != -1):
            pca = PCA(n_components = n_com) #or you can choose the n_components
            principalComponents = pca.fit_transform(df) 
            df = pd.DataFrame(data = principalComponents)
            self.df = df
            return df
        else:
            pca = PCA(n_components = 30) #or you can choose the n_components
            principalComponents = pca.fit_transform(df)
            cumsum = pca.explained_variance_ratio_.cumsum()
            cumsum_momentum = []
            for i in range(len(cumsum)-1):
                tmp = (cumsum[i+1]-cumsum[i])/cumsum[i]
                cumsum_momentum.append(tmp)
            if (plot == 1):
                plt.plot(cumsum)
                plt.title("cumulative variance sum")
                plt.show()
                plt.plot(cumsum_momentum)
                plt.title("cumulative variance sum momentum")
                plt.show()
            n_com = 2
            for i in range (2, 30):
                if (cumsum[i] >=0.95 and cumsum_momentum[i] <=0.05):
                    print("Your cumulative variance sum and its momentum are: ", cumsum[i], cumsum_momentum[i])
                    break
                else:
                    n_com+=1
            print("The default n_com is ", n_com)
            if (n_com >= 30):
                print("Warning: you should use a higher trial n_components rather than 30")
            pca = PCA(n_components = n_com) #or you can choose the n_components
            principalComponents = pca.fit_transform(df) 
            df = pd.DataFrame(data = principalComponents)
            self.df = df
            return df
    
    
    def rescaling(self): #(x-min)/(max-min) save training time
        print("Recaling")
        df = self.df
        for i in df.index:
            cur_max = np.max(np.abs(df.loc[i]))
            cur_min = np.min(np.abs(df.loc[i]))
            for j in range(1,len(df.loc[i])):
                if (df.loc[i][j] != 0):
                    sign = df.lo[i][j]/np.abs(df.loc[i][j])
                    df.loc[i][j] = (np.abs(df.loc[i][j])-cur_min)/(cur_max-cur_min)*sign
                else:
                    df.loc[i][j] = (np.abs(df.loc[i][j])-cur_min)/(cur_max-cur_min)
        self.df = df
        return df
    
    def reset(self):
        x_data, y_data = extract_x_y(self.file)
        x_data = pd.DataFrame(x_data)
        self.brix = y_data
        self.df = x_data
        self.col = self.df.columns
        self.row = self.df.index