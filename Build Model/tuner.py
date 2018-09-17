import numpy as np
import random
from datetime import datetime
import pandas as pd
import copy



######################################################################################################
#This is tuning box embedded in each climber
#The tuning box decodes the tuning code and order climbers to move directly
######################################################################################################
class Tuner():
    def __init__(self, x_data, y_data, tuning_code = {
            "MAS_windows_size": -1, "DCT_typo": -1, "PCA_n_com": -1, 
                           "SVR_rbf_gamma": -1, "SVR_rbf_c": -1, "SVR_rbf_epsilon": -1,
                           "SVR_poly_gamma": -1, "SVR_poly_c": -1, "SVR_poly_epsilon": -1,
                           "SVR_poly_coef0": -1, "SVR_poly_degree": -1, "PLS_n_components": -1},
    last_progress = 0, last_tuning_code = {
            "MAS_windows_size": -1, "DCT_typo": -1, "PCA_n_com": -1, 
                           "SVR_rbf_gamma": -1, "SVR_rbf_c": -1, "SVR_rbf_epsilon": -1,
                           "SVR_poly_gamma": -1, "SVR_poly_c": -1, "SVR_poly_epsilon": -1,
                           "SVR_poly_coef0": -1, "SVR_poly_degree": -1, "PLS_n_components": -1}, 
    last_values = {
            "MAS_windows_size": 3, "DCT_typo": 1, "PCA_n_com": -1, 
                           "SVR_rbf_gamma": 8.99e-05, "SVR_rbf_c": 1000, "SVR_rbf_epsilon": 4.8e-06,
                           "SVR_poly_gamma": 3.21e-05, "SVR_poly_c": 6100, "SVR_poly_epsilon": 5.31e-05,
                           "SVR_poly_coef0": 1.6000001, "SVR_poly_degree": 2, "PLS_n_components": 11}):
        #default self.tuning_code = {"MAS_windows_size": -1, "DCT_typo": -1, "PCA_n_com": -1, 
#                           "SVR_rbf_gamma": -1, "SVR_rbf_c": -1, "SVR_rbf_epsilon": -1,
#                           "SVR_poly_gamma": -1, "SVR_poly_c": -1, "SVR_poly_epsilon": -1,
#                           "SVR_poly_coef0": -1, "SVR_poly_degree": -1, "PLS_n_components": -1}
        self.tuning_code = tuning_code
        self.last_tuning_code = last_tuning_code
        self.last_progress = abs(last_progress)
        
        self.MAS_windows_size = 3
        

        self.DCT_typo = last_values["DCT_typo"]
        
        self.PCA_n_com = last_values["PCA_n_com"]
        
        self.SVR_rbf_gamma = last_values["SVR_rbf_gamma"]
        self.SVR_rbf_c = last_values["SVR_rbf_c"]
        self.SVR_rbf_epsilon = last_values["SVR_rbf_epsilon"]
        
        self.SVR_poly_gamma = last_values["SVR_poly_gamma"]
        self.SVR_poly_c = last_values["SVR_poly_c"]
        self.SVR_poly_epsilon = last_values["SVR_poly_epsilon"]
        self.SVR_poly_coef0 = last_values["SVR_poly_coef0"]
        self.SVR_poly_degree = last_values["SVR_poly_degree"]
        
        self.PLS_n_components = last_values["PLS_n_components"]
        
        self.x_data = pd.DataFrame(x_data)
        self.y_data = y_data
        
        self.last_values = last_values
        
        
    def random_distribute_values(self):
        mu, sigma = 0, 2 # mean and standard deviation
        last_values = self.last_values
        random.seed(datetime.now())
        
        self.MAS_windows_size = random.sample(list(np.arange(1, (len(self.x_data.columns)), 1)), 1)[0]
        
        self.DCT_typo = random.sample([1, 2, 3], 1)[0]
        
        self.PCA_n_com = random.sample(list(np.arange(3, 30, 1)), 1)[0]
        
        self.SVR_rbf_gamma = last_values["SVR_rbf_gamma"] * abs(np.random.normal(mu, sigma, 1))[0]
        self.SVR_rbf_c = last_values["SVR_rbf_c"] * abs(np.random.normal(mu, sigma, 1))[0]
        self.SVR_rbf_epsilon = last_values["SVR_rbf_epsilon"] * abs(np.random.normal(mu, sigma, 1))[0]
        
        self.SVR_poly_gamma = last_values["SVR_poly_gamma"] * abs(np.random.normal(mu, sigma, 1))[0]
        self.SVR_poly_c = last_values["SVR_poly_c"] * abs(np.random.normal(mu, sigma, 1))[0]
        self.SVR_poly_epsilon = last_values["SVR_poly_epsilon"] * abs(np.random.normal(mu, sigma, 1))[0]
        self.SVR_poly_coef0 = last_values["SVR_poly_coef0"] * abs(np.random.normal(mu, sigma, 1))[0]
        self.SVR_poly_degree = random.sample(list(np.arange(1, 5, 1)), 1)[0]
        
        self.PLS_n_components = random.sample(list(np.arange(1, 30, 1)), 1)[0]
        
        
        
        self.last_values["MAS_windows_size"] = self.MAS_windows_size
        
        self.last_values["DCT_typo"] = self.DCT_typo 
        
        self.last_values["PCA_n_com"] = self.PCA_n_com
        
        self.last_values["SVR_rbf_gamma"] = self.SVR_rbf_gamma
        self.last_values["SVR_rbf_c"] = self.SVR_rbf_c
        self.last_values["SVR_rbf_epsilon"] = self.SVR_rbf_epsilon
        
        self.last_values["SVR_poly_gamma"] = self.SVR_poly_gamma
        self.last_values["SVR_poly_c"] = self.SVR_poly_c
        self.last_values["SVR_poly_epsilon"] = self.SVR_poly_epsilon
        self.last_values["SVR_poly_coef0"] = self.SVR_poly_coef0
        self.last_values["SVR_poly_degree"] = self.SVR_poly_degree
        
        self.last_values["PLS_n_components"] = self.PLS_n_components
        
        return last_values
        
        
        
    def tuning_hotpot(self, tuning_code, last_progress = 0, lr_shrink = 10):

#        print("From: ", self.SVR_rbf_gamma, self.SVR_rbf_c, self.SVR_rbf_epsilon, self.PCA_n_com)
        if tuning_code['MAS_windows_size'] == -1:
            pass
        elif tuning_code['MAS_windows_size'] == 0:
            if self.MAS_windows_size > 2 and self.MAS_windows_size < int(len(self.x_data.columns)/2):
                self.MAS_windows_size -= 1
            elif self.MAS_windows_size <= 2:
#                print("Warning: MAS_windows_size reached the minimum value 2")
                self.MAS_windows_size = 2
                tuning_code['MAS_windows_size'] = 1
        elif tuning_code['MAS_windows_size'] == 1:
            if self.MAS_windows_size > 2 and self.MAS_windows_size < int(len(self.x_data.columns))/2:
                self.MAS_windows_size += 1
            elif self.MAS_windows_size >= int(len(self.x_data.columns))/2:
#                print("Warning: MAS_windows_size reached the maximum value int(len(self.x_data.columns))/2")
                self.MAS_windows_size = int(len(self.x_data.columns))/2
                tuning_code['MAS_windows_size'] = 0
    
        if tuning_code['DCT_typo'] == -1:
            pass
        elif tuning_code['DCT_typo'] == 0:
            if self.DCT_typo >= 2:
                self.DCT_typo -= 1
            elif self.DCT_typo < 2:
                self.DCT_typo = 1
                tuning_code['DCT_typo'] = 1
        elif tuning_code['DCT_typo'] == 1:
            if self.DCT_typo <= 2:
                self.DCT_typo +=1
            if self.DCT_typo > 2:
                self.DCT_typo = 3
                tuning_code['DCT_typo'] = 0
                
                
        if tuning_code['PCA_n_com'] == -1:
            pass
        elif tuning_code['PCA_n_com'] == 0:
            if self.PCA_n_com > len(self.x_data.columns):
                self.PCA_n_com = len(self.x_data.columns)
                tuning_code['PCA_n_com'] = 0
            if self.PCA_n_com > 3:
                self.PCA_n_com -= 1
            elif self.PCA_n_com <= 3:
                self.PCA_n_com = 3
                tuning_code['PCA_n_com'] = 1
        elif tuning_code['PCA_n_com'] == 1:
            if self.PCA_n_com < 3:
                self.PCA_n_com = 3
                tuning_code['PCA_n_com'] = 1
            if self.PCA_n_com < len(self.x_data.columns):
                self.PCA_n_com +=1
            elif self.PCA_n_com > len(self.x_data.columns):
                self.PCA_n_com = len(self.x_data.columns)
                tuning_code['PCA_n_com'] = 0
    
        lr = 1/(1+np.exp(self.last_progress))/lr_shrink
        
        
        if tuning_code['SVR_rbf_gamma'] == -1:
            pass
        elif tuning_code['SVR_rbf_gamma'] == 0:
            self.SVR_rbf_gamma *= (1-lr)
        elif tuning_code['SVR_rbf_gamma'] == 1:
            self.SVR_rbf_gamma *= (1+lr)
            
        if tuning_code['SVR_rbf_c'] == -1:
            pass
        elif tuning_code['SVR_rbf_c'] == 0:
            self.SVR_rbf_c *= (1-lr)
        elif tuning_code['SVR_rbf_c'] == 1:
            self.SVR_rbf_c *= (1+lr)
            
        if tuning_code['SVR_rbf_epsilon'] == -1:
            pass
        elif tuning_code['SVR_rbf_epsilon'] == 0:
            self.SVR_rbf_epsilon *= (1-lr)
        elif tuning_code['SVR_rbf_epsilon'] == 1:
            self.SVR_rbf_epsilon *= (1+lr)
            
        if tuning_code['SVR_poly_gamma'] == -1:
            pass
        elif tuning_code['SVR_poly_gamma'] == 0:
            self.SVR_poly_gamma *= (1-lr)
        elif tuning_code['SVR_poly_gamma'] == 1:
            self.SVR_poly_gamma *= (1+lr)
            
        if tuning_code['SVR_poly_c'] == -1:
            pass
        elif tuning_code['SVR_poly_c'] == 0:
            self.SVR_poly_c *= (1-lr)
        elif tuning_code['SVR_poly_c'] == 1:
            self.SVR_poly_c *= (1+lr)
            
        if tuning_code['SVR_poly_epsilon'] == -1:
            pass
        elif tuning_code['SVR_poly_epsilon'] == 0:
            self.SVR_poly_epsilon *= (1-lr)
        elif tuning_code['SVR_poly_epsilon'] == 1:
            self.SVR_poly_epsilon *= (1+lr)
            
        if tuning_code['SVR_poly_coef0'] == -1:
            pass
        elif tuning_code['SVR_poly_coef0'] == 0:
            self.SVR_poly_coef0 *= (1-lr)
        elif tuning_code['SVR_poly_coef0'] == 1:
            self.SVR_poly_coef0 *= (1+lr)
            
        if tuning_code['SVR_poly_degree'] == -1:
            pass
        elif tuning_code['SVR_poly_degree'] == 0:
            if (self.SVR_poly_degree > 1):
                self.SVR_poly_degree -= 1
            else:
                self.SVR_poly_degree = 1
                tuning_code['SVR_poly_degree'] = 1
        elif tuning_code['SVR_poly_degree'] == 1:
            self.SVR_poly_degree += 1         
    
    
        if tuning_code['PLS_n_components'] == -1:
            pass
        elif tuning_code['PLS_n_components'] == 0:
            if (self.PLS_n_components > 1):
                self.PLS_n_components -= 1
            else:
                self.PLS_n_components = 1
                tuning_code['PLS_n_components'] = 1
        elif tuning_code['PLS_n_components'] == 1:
            self.PLS_n_components += 1         
            
            
        self.last_values["MAS_windows_size"] = self.MAS_windows_size
        
        self.last_values["DCT_typo"] = self.DCT_typo 
        
        self.last_values["PCA_n_com"] = self.PCA_n_com
        
        self.last_values["SVR_rbf_gamma"] = self.SVR_rbf_gamma
        self.last_values["SVR_rbf_c"] = self.SVR_rbf_c
        self.last_values["SVR_rbf_epsilon"] = self.SVR_rbf_epsilon
        
        self.last_values["SVR_poly_gamma"] = self.SVR_poly_gamma
        self.last_values["SVR_poly_c"] = self.SVR_poly_c
        self.last_values["SVR_poly_epsilon"] = self.SVR_poly_epsilon
        self.last_values["SVR_poly_coef0"] = self.SVR_poly_coef0
        self.last_values["SVR_poly_degree"] = self.SVR_poly_degree
        
        self.last_values["PLS_n_components"] = self.PLS_n_components
        
        
#        print("To: ", self.SVR_rbf_gamma, self.SVR_rbf_c, self.SVR_rbf_epsilon, self.PCA_n_com)
        return copy.deepcopy(self.last_values)
    
    
    
    
    
    
    
