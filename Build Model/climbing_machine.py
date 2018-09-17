from tuner import Tuner
import random
from datetime import datetime
from combinations import Combinations
import numpy as np
import copy

from preprocessing_library import Preprocessing


######################################################################################################
#A class to design the features of climbers
#tuning code: the order for the next step
#location: a dictionary to record the value of each key parameters
#prepro_index: preprocessing methods used in the model
#discriminant_index: discriminant methods used int the model
#map_variation: show the instability of a model
#last_height: the accuracy the climber achieved at the last turn of training period
#height: the accuracy the climber achieved at the current training period
#x_data: the matrix of independent variables
#y_data: the matrix of dependent variables
######################################################################################################


class Climbing_machine(Tuner, Combinations):
    def __init__(self, x_data, y_data, tuning_code, location, prepro_index, discriminant_index,
                 map_variation, input_types = 2, prepro_types = 7, verbose = 0):
        self.tuning_code = tuning_code
        self.last_tuning_code = {"MAS_windows_size": -1, "DCT_typo": -1, "PCA_n_com": -1, 
                           "SVR_rbf_gamma": -1, "SVR_rbf_c": -1, "SVR_rbf_epsilon": -1,
                           "SVR_poly_gamma": -1, "SVR_poly_c": -1, "SVR_poly_epsilon": -1,
                           "SVR_poly_coef0": -1, "SVR_poly_degree": -1, "PLS_n_components": -1}
        self.location = location
        self.last_height = 0
        self.height = 0
        self.map_variation = map_variation
        self.x_data = x_data
        self.y_data = y_data
        self.prepro_index = prepro_index
        self.discriminant_index = discriminant_index
        self.n_step = 0
        self.n_detectors = 0
        self.input_types = input_types
        self.prepro_types = prepro_types
        self.verbose = verbose
        Tuner.__init__(self, x_data = x_data, y_data = y_data, 
                       tuning_code = tuning_code, last_values = location)
        Combinations.__init__(self, input_types, prepro_types)
        
        
######################################################################################################
#measure and return the height of the current location
######################################################################################################
    def measure_height(self):
        print("Measuring height/re-applying preprocessing and discriminant methods after tuning parameters")
        self.x_data, self.y_data = Combinations.prepro_decode(self, self.prepro_index, self.x_data, self.y_data)
        self.height = self.discriminant_decode(self.discriminant_index, self.x_data, self.y_data,
                                               self.location)
        return self.height
    
    
######################################################################################################
#return the number of detectors in the climber
######################################################################################################
    def num_detectors(self, tuning_code):
        detectors = 0
        for param in list(tuning_code):
            if tuning_code[param] != -1:
                detectors += 1
        return detectors
        
    
    
######################################################################################################
#randomly initialize the tuning code of a climber
######################################################################################################
    def tuning_code_initialization(self, prepro_index, discriminant_index):
        self.stay_still()
        for i in prepro_index:
            random.seed(datetime.now())
            if i == 1:
                self.tuning_code["MAS_windows_size"] = random.sample([0, 1], 1)[0]
            elif i == 2:
                self.tuning_code["DCT_typo"] = random.sample([0, 1], 1)[0]
            elif i == 3:
                self.tuning_code["PCA_n_com"] = random.sample([0, 1], 1)[0]
        if (discriminant_index == 1):
            self.tuning_code["SVR_rbf_gamma"] = random.sample([0, 1], 1)[0]
            self.tuning_code["SVR_rbf_c"] = random.sample([0, 1], 1)[0]
            self.tuning_code["SVR_rbf_epsilon"] = random.sample([0, 1], 1)[0]
        elif (discriminant_index == 2):
            self.tuning_code["SVR_poly_gamma"] = random.sample([0, 1], 1)[0]
            self.tuning_code["SVR_poly_c"] = random.sample([0, 1], 1)[0]
            self.tuning_code["SVR_poly_epsilon"] = random.sample([0, 1], 1)[0]
            self.tuning_code["SVR_poly_coef0"] = random.sample([0, 1], 1)[0]
            self.tuning_code["SVR_poly_degree"] = random.sample([0, 1], 1)[0]
        elif (discriminant_index == 4):
            self.tuning_code["PLS_n_components"] = random.sample([0, 1], 1)[0]
            
        if (self.n_step == 1):
            self.detectors = self.num_detectors(self.tuning_code)
                        
        return self.tuning_code
        
    
    
######################################################################################################
#order the climber to stay still
######################################################################################################
    def stay_still(self):
        self.tuning_code = {"MAS_windows_size": -1, "DCT_typo": -1, "PCA_n_com": -1, 
                           "SVR_rbf_gamma": -1, "SVR_rbf_c": -1, "SVR_rbf_epsilon": -1,
                           "SVR_poly_gamma": -1, "SVR_poly_c": -1, "SVR_poly_epsilon": -1,
                           "SVR_poly_coef0": -1, "SVR_poly_degree": -1, "PLS_n_components": -1}


######################################################################################################
#locate the climber at a random location
######################################################################################################
    def locate2_random_location(self):
        self.location = self.random_distribute_values()
        self.height = self.measure_height()
        return self.location
        
    
######################################################################################################
#return all possible tuning code on the available dimensions
######################################################################################################
    def detection360_all_tuning_code(self, initialized_tuning_code, binary_switchOn_list):
        param_list = []
        all_param = list(initialized_tuning_code)
        for param in all_param:
            if initialized_tuning_code[param] != -1:
                param_list.append(param)
            
        tuning_code_list = []
        for binary in binary_switchOn_list:
            new_tuning_code = initialized_tuning_code
            counter = 0
            for i in range(len(binary)):
                new_tuning_code[param_list[counter]] = binary[i]
                counter += 1
            tuning_code_list.append(copy.deepcopy(new_tuning_code))
        
        return tuning_code_list
    
    
    
    def binary2_full_string(self, binary, full_len):
        len_diff = full_len-len(binary)
        return '0'*len_diff + binary 
    

######################################################################################################
#order climbers to move one step by following the tuning code.
#mode: the current movement mode
#last_progress: the improvement made from the last training period. It is used to une the learning rate
######################################################################################################
    def move_one_step(self, mode, last_progress):
        #available mode: 360_detection, random_detection, random_movement
        self.n_step += 1
        self.last_progress = last_progress
        random.seed(datetime.now())
        if mode == 'random_movement':
            self.tuning_code = self.tuning_code_initialization(self.prepro_index, self.discriminant_index)
            new_location = self.tuning_hotpot(self.tuning_code, self.height-self.last_height)
            self.last_tuning_code = self.tuning_code
            self.location = new_location
#            print(self.location['SVR_rbf_gamma'], self.location['SVR_rbf_c'],
#                  self.location['SVR_rbf_epsilon'], self.location['PCA_n_com'])

            self.last_height = self.height
            self.height = self.measure_height()
            return self.height, copy.deepcopy(self.location)
            
        elif mode == '360_detection':
            if self.n_detectors >3:
                print("Warning: too many detectors! Recommend using less than 5 detectors under 360_detection mode")
            max_binary = 2**self.n_detectors
            binary_switchOn_list = []
            for i in np.arange(0, max_binary, 1):
                new_binary = bin(i)[2:]
                new_binary = self.binary2_full_string(new_binary, self.n_detectors)
                binary_switchOn_list.append(new_binary)
            tuning_code_list = self.detection360_all_tuning_code(self.tuning_code, binary_switchOn_list)
            height_list = []
            for code in tuning_code_list:
                new_location = self.tuning_hotpot(self.tuning_code)
                self.last_tuning_code = self.tuning_code
                self.location = new_location
                self.height = self.measure_height()
                height_list.append(self.height)
            max_height = max(height_list)
            optimal_code = tuning_code_list[np.argmax(height_list)]
            new_location = self.tuning_hotpot(optimal_code, self.last_progress)
            self.last_tuning_code = optimal_code
            self.location = new_location
            self.last_height = self.height
            self.height = max_height
            
            return self.height, copy.deepcopy(self.location)
            #print("Move to location : ", new_location, " with current max height : ", max_height)
            
        elif mode == 'random_detection':
            ## try to repeat the last movement first
            old_location = self.location
            if (self.n_step == 1):
                self.height = self.measure_height()
            self.last_height = self.height
            new_location = self.tuning_hotpot(self.last_tuning_code)
            if (self.height == 0):
                self.height = self.measure_height()
            if (self.height - self.last_height) < self.map_variation:
                self.location = old_location
                self.height = self.last_height
                max_binary = 2**self.n_detectors
                binary_switchOn_list = []
                for i in np.arange(0, max_binary, 1):
                    new_binary = bin(i)[2:]
                    new_binary = self.binary2_full_string(new_binary, self.n_detectors)
                    binary_switchOn_list.append(new_binary)
                tuning_code_list = self.detection360_all_tuning_code(self.tuning_code, binary_switchOn_list)
                for _ in range(max_binary):
                    random.seed(datetime.now())
                    random_tuning_code = random.sample(tuning_code_list, 1)
                    if random_tuning_code in tuning_code_list: tuning_code_list.remove(random_tuning_code)
                    new_location = self.tuning_hotpot(self.tuning_code)
                    self.last_tuning_code = self.tuning_code
                    self.location = new_location
                    self.last_height = self.height
                    self.height = self.measure_height()
                    if (self.height - self.last_height) > self.map_variation:
                        if self.verbose == 2: print("from height ", self.last_height, " to ", self.height)
                        self.last_progress = self.height - self.last_height
                        return self.height, copy.deepcopy(self.location)
                    if self.verbose == 2: print("Locating on a flat area. Machine is moving randomly")
            return self.height, copy.deepcopy(self.location)
                
            

            
            
                
######################################################################################################
#the following code is used to test this module
######################################################################################################
                
#file = "Referenced Ultimate Oranges Matrix.xlsx"
#preprocessing = Preprocessing(file)
#x_data, y_data = preprocessing.extract_x_y(file)
#tuner = Tuner(x_data, y_data)
#tuning_code = tuner.tuning_code
#location = tuner.last_values
#prepro_index = 'r'
#discriminant_index = 1
#map_variation = 0.025
#climbing_machine = Climbing_machine (x_data, y_data, tuning_code, location, prepro_index, discriminant_index,
#                 map_variation)
#climbing_machine.move_one_step("random_detection")             
#                
#                
#                
                
                
                
                
            
