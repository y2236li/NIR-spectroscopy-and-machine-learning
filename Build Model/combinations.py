from Find_all_combos import All_combos
from Inputs import Inputs
import pandas as pd
import numpy as np
from preprocessing_library import Preprocessing
from discriminant_functions import Discriminant_functions
import sys
from tuner import Tuner


######################################################################################################
#A tool box to find all possible combinations of input and preprocessing methods
######################################################################################################


class Combinations(Inputs):
    def __init__(self, input_types, prepro_types):
        self.referenced_file = "Referenced Ultimate Oranges Matrix.xlsx"
        self.unreferenced_file = "Unreferenced Ultimate Oranges Matrix.xlsx"
        self.input_types = input_types
        self.prepro_types = prepro_types
        
    def input_code(self):
        all_combos = All_combos(self.input_types)
        combos_index_r = all_combos.index_result()
        combos_index_ur = all_combos.index_result()
        for i in range(len(combos_index_r)):
            combos_index_r[i] = 'r' + combos_index_r[i]
            combos_index_ur[i] = 'u' + combos_index_ur[i]
        combos_index_r.append('r')
        combos_index_ur.append('u')
        self.combos_index_r = set(combos_index_r)
        self.combos_index_ur = set(combos_index_ur)
        return list(set(combos_index_r + combos_index_ur))
        
    def input_decode(self, code):
        df = pd.DataFrame()
        referenced = False
        for c in code:
            if (c == 'r'):
                print("Referenced data matrix")
                x_data, y_data, colnames = self.referenced()
                referenced = True
                df = pd.DataFrame(x_data)
            elif (c == 'u'):
                print("Unreferenced data matrix")
                x_data, y_data, colnames = self.unreferenced()
                referenced = False
                df = pd.DataFrame(x_data)
            else:
                print("Adding ", np.int(c)+1, "order derivative into matrix")
                new_x_data, y_data = self.derivative(order = np.int(c)+1, referenced = referenced)
                new_df = pd.DataFrame(new_x_data)
                df = pd.concat([df, new_df], axis=1, ignore_index=True)
        return df, y_data
    
    
    
    
    def sort_str(self, strNum):
        num_list = []
        for i in strNum:
            num_list.append(int(i))
        num_list.sort()
        return num_list
    
    
    def index_result(self, methods_list, building_list, max_len):
        new_building_list = []
        for building_item in building_list:
            for methods_item in methods_list:
                if not (str(methods_item) in str(building_item)):
                    new_building_list.append(str(building_item) + str(methods_item))
        max_len -= 1
        if (max_len > 1):
            return self.index_result(methods_list, new_building_list, max_len)
        else:
            num_new_building_list = []
            for i in range(len(new_building_list)):
                for j in range(len(new_building_list[i])):
                    sort = self.sort_str(new_building_list[i])
                    if not sort in num_new_building_list:
                        num_new_building_list.append(sort)
            return num_new_building_list
    
    def prepro_code(self, max_len = 2):
        methods_list = np.arange(1, self.prepro_types+1, 1)
        building_list = methods_list
        if max_len > 1:
            index = self.index_result(methods_list, building_list, max_len)
            for i in np.arange(1, max_len, 1):
                if i > 1:
                    tmp_index = self.index_result(methods_list, building_list, i)
                elif i == 1:
                    tmp_index = list(methods_list)
                index += tmp_index
        elif max_len == 1:
            return methods_list
        else:
            sys.exit("ERROR: Please use at least one preprocessing methods!")

        return index
    
    def prepro_decode(self, code, x_data, y_data,
                      tuning_code = {"MAS_windows_size": -1, "DCT_typo": -1, "PCA_n_com": -1, 
                           "SVR_rbf_gamma": -1, "SVR_rbf_c": -1, "SVR_rbf_epsilon": -1,
                           "SVR_poly_gamma": -1, "SVR_poly_c": -1, "SVR_poly_epsilon": -1,
                           "SVR_poly_coef0": -1, "SVR_poly_degree": -1, "PLS_n_components": -1}):
        #tuning_code = {"windows_size": -1, "typo": -1, "n_com": -1}
        #-1 means default value
        prepro = Preprocessing(self.referenced_file)
        prepro.df = x_data
        prepro.brix = y_data
        prepro.col = x_data.columns
        prepro.row = x_data.index
        tuner = Tuner(x_data, y_data, tuning_code)
        self.last_value = tuner.tuning_hotpot(tuning_code)
        for i in code:
            if i == 1:
                prepro.Moving_average_smoothing(windows_size = tuner.MAS_windows_size)
            elif i == 2:
                prepro.rescaling()
            elif i == 3:
                prepro.PCA(n_com = tuner.PCA_n_com) ## n_component
            elif i == 4:
                prepro.SNV()
            elif i == 5:
                prepro.MSC()
            elif i == 6:
                prepro.detrend()
            elif i == 7:
                prepro.DCT(typo=tuner.DCT_typo) ## type

        return prepro.df, prepro.brix
    
    
    
    def discriminant_decode(self, code, x_data, y_data,
                      tuning_code = {"MAS_windows_size": -1, "DCT_typo": -1, "PCA_n_com": -1, 
                           "SVR_rbf_gamma": -1, "SVR_rbf_c": -1, "SVR_rbf_epsilon": -1,
                           "SVR_poly_gamma": -1, "SVR_poly_c": -1, "SVR_poly_epsilon": -1,
                           "SVR_poly_coef0": -1, "SVR_poly_degree": -1, "PLS_n_components": -1}):
        if (code > 4):
            sys.exit("Error: the system is base on one discriminant function. Received more than one function request!")
        
        tuner = Tuner(x_data = x_data, y_data = y_data, tuning_code = tuning_code)
        discriminant_functions = Discriminant_functions()
        
        result = -1
        
        if (code == 1):
            
            try:
                result = discriminant_functions.svr_CV(x_data, y_data, gamma = tuner.SVR_rbf_gamma,
                                          epsilon = tuner.SVR_rbf_epsilon, c = tuner.SVR_rbf_c)
            except:
                pass
            
        elif (code == 2):
            try:
                result = discriminant_functions.svr_CV_poly(x_data, y_data, gamma = tuner.SVR_poly_gamma,
                                          epsilon = tuner.SVR_poly_epsilon, c = tuner.SVR_poly_c,
                                          coef0 = tuner.SVR_poly_coef0, degree = tuner.SVR_poly_degree)
            except:
                pass
        
        elif (code == 3):
            try:
                result =discriminant_functions.decision_tree_CV(x_data, y_data)
                
            except:
                pass
            
        elif (code == 4):
            try:
                result = discriminant_functions.pls_CV(x_data, y_data, tuner.PLS_n_components)
            
            except:
                pass
            
        return result
        
        
        
            
        
        
                
#input_types = 2
#prepro_types = 7
#combinations = Combinations(input_types, prepro_types)
#input_combos = combinations.input_code()
#prepro_combos = combinations.prepro_code(max_len = 4)
#x_data, y_data = combinations.input_decode(input_combos[1])
#x_data, y_data = combinations.prepro_decode(prepro_combos[1], x_data, y_data)
