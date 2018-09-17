import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing_library import Preprocessing
from ML_functions import svr_CV


    
    
class Test(Preprocessing):
    def __init__ (self, file):
        Preprocessing.__init__ (self, file)
    
    def test_moving_average_smoothing(self):
        window_size_list = np.arange(1, int(len(self.row)/2), 1)
        centered_list = [0, 1]
        for j in centered_list:
            accuracy_list = []
            for i in window_size_list:
                self.Moving_average_smoothing(center = 0, windows_size = 14)
                accuracy = svr_CV(self.df, self.brix)
                accuracy_list.append(accuracy)
                self.reset()
            plt.plot(accuracy_list)
            plt.show()
        print("Maximum: ", window_size_list[np.argmax(accuracy_list)])
            
            
    def test_MSC(self):
        accuracy_list = []
        print("Without MSC")
        for _ in range(20):
            accuracy = svr_CV(self.df, self.brix)
            accuracy_list.append(accuracy)
        plt.plot(accuracy_list)
        plt.show()
        accuracy_list = []
        
        
        print("Applying MSC")
        self.MSC()
        for _ in range(20):
            accuracy = svr_CV(self.df, self.brix)
            accuracy_list.append(accuracy)
        self.reset()
        plt.plot(accuracy_list)
        plt.show()
        
        
        
    def test_detrend(self): 
        accuracy_list = []
        print("Applying detrend")
        self.detrend()
        for _ in range(20):
            accuracy = svr_CV(self.df, self.brix)
            accuracy_list.append(accuracy)
        self.reset()
        plt.plot(accuracy_list)
        plt.show()
    
    

    
    def test_SNV(self):
        accuracy_list = []
        print("Applying SNV")
        self.SNV()
        for _ in range(20):
            accuracy = svr_CV(self.df, self.brix)
            accuracy_list.append(accuracy)
        self.reset()
        plt.plot(accuracy_list)
        plt.show()
        
        
    def test_DCT(self):
        accuracy_list = []
        print("Applying DCT type 1")
        self.DCT(1)
        for _ in range(20):
            accuracy = svr_CV(self.df, self.brix)
            accuracy_list.append(accuracy)
        self.reset()
        plt.plot(accuracy_list)
        plt.show()
        
        accuracy_list = []
        print("Applying DCT type 2")
        self.DCT(2)
        for _ in range(20):
            accuracy = svr_CV(self.df, self.brix)
            accuracy_list.append(accuracy)
        self.reset()
        plt.plot(accuracy_list)
        plt.show()
        
        accuracy_list = []
        print("Applying DCT type 3")
        self.DCT(3)
        for _ in range(20):
            accuracy = svr_CV(self.df, self.brix)
            accuracy_list.append(accuracy)
        self.reset()
        plt.plot(accuracy_list)
        plt.show()
        
    def test_PCA(self):
        accuracy_list = []
        print("Applying PCA")
        n_com_list = np.arange(1, int(len(self.row)/2), 1)
        for i in n_com_list:
            self.PCA(n_com = i)
            accuracy = svr_CV(self.df, self.brix)
            accuracy_list.append(accuracy)
            self.reset()
        plt.plot(accuracy_list)
        plt.show()
        print("Maximum: ", n_com_list[np.argmax(accuracy_list)])
        
    def test_rescaling(self):
        accuracy_list = []
        print("Applying rescaling")
        self.rescaling()
        for _ in range(20):
            accuracy = svr_CV(self.df, self.brix)
            accuracy_list.append(accuracy)
        self.reset()
        plt.plot(accuracy_list)
        plt.show()
        
    def test_pure_SVR(self, time):
        accuracy_list = []
        print("Running SVR only")
        for _ in range(time):
            accuracy = svr_CV(self.df, self.brix)
            accuracy_list.append(accuracy)
        plt.plot(accuracy_list)
        plt.show()



file = 'Referenced Ultimate Oranges Matrix.xlsx'

test = Test(file)
#test.test_moving_average_smoothing()
test.test_pure_SVR(200)
#test.test_DCT()
#test.test_PCA()





    
    
    