from combinations import Combinations
import numpy as np
import random
from datetime import datetime
import scipy.stats
from climbing_machine import Climbing_machine
import matplotlib.pyplot as plt
import copy
import pandas as pd


######################################################################################################
#The default data set are "Referenced Ultimate Oranges Matrix.xlsx" and "Unreferenced Ultimate Oranges Matrix.xlsx"
#input_types: number of input types
#prepro_types: number of preprocessing method types
#dis_types: number of discriminant method types
#max_prepro: the maximum number of preprocessing method being applied before precessing
#referenced_file: xlsx file of referenced data
#unreferennced_file: xlsx file of unreferenced data
#variation_strength: a quantified value indicating the instability of the prediction from a specific machine learning model
#detectors: number of detectors on climbers
#climbing_machine_list: save a list of class of climbers
#dimension_name: the name of key parameters in a model
#verbose: showing the details when model is training
#best_climbers_list: save the climbers made the top performance
#default_tuning_code: order climbers to stay still
#default_loctaion: order climbers locate at a specific position
######################################################################################################

class Climbing_machine_configuration(Combinations):
    def __init__(self, max_prepro = 4, referenced_file = "Referenced Ultimate Oranges Matrix.xlsx",
                 unreferenced_file = "Unreferenced Ultimate Oranges Matrix.xlsx",
                 input_types = 2, prepro_types = 7, dis_types = 4):
        self.max_prepro = max_prepro
        self.referenced_file = referenced_file
        self.unreferenced_file = unreferenced_file
        self.input_types = 2
        self.prepro_types = 7
        self.dis_types = 4
        self.variation_strength = -1
        self.detectors = 0
        self.climbing_machine_list = []
        self.accuracy_list = []
        self.dimension_name = []
        self.verbose = 0
        self.best_climbers_list = []
        Combinations.__init__(self, input_types, prepro_types)
        self.default_tuning_code = {"MAS_windows_size": -1, "DCT_typo": -1, "PCA_n_com": -1, 
                           "SVR_rbf_gamma": -1, "SVR_rbf_c": -1, "SVR_rbf_epsilon": -1,
                           "SVR_poly_gamma": -1, "SVR_poly_c": -1, "SVR_poly_epsilon": -1,
                           "SVR_poly_coef0": -1, "SVR_poly_degree": -1, "PLS_n_components": -1}
        self.default_location = {
            "MAS_windows_size": 3, "DCT_typo": 1, "PCA_n_com": -1, 
                           "SVR_rbf_gamma": 8.99e-05, "SVR_rbf_c": 1000, "SVR_rbf_epsilon": 4.8e-06,
                           "SVR_poly_gamma": 3.21e-05, "SVR_poly_c": 6100, "SVR_poly_epsilon": 5.31e-05,
                           "SVR_poly_coef0": 1.6000001, "SVR_poly_degree": 2, "PLS_n_components": 11}


######################################################################################################
#return the all possible combinations of input and perprocesssing method in terms of code. The code is
#from 0 to n-1, where n is the number of types. Each code number represent each type of function
######################################################################################################
        
    def set_combinations(self, input_types = -1, prepro_types = -1):
        if input_types != -1:
            self.input_types == input_types
        if prepro_types != -1:
            self.prepro_types == prepro_types
        self.input_combos = Combinations.input_code(self)
        self.prepro_combos = Combinations.prepro_code(self, max_len = self.max_prepro)
 
 
 
 
######################################################################################################
#Calculate conficence inteval to represent the instability of a model.
#Return peak to peak value as conficence range/variation strength
######################################################################################################
    def Variation_strength(self, x_data, y_data, discriminant_index, test_period = 20, confidence = 0.95):
        if self.verbose == 2: print("Testing map variation strength by confidence index ", confidence)
        accuracy_list = []
        for _ in range(test_period):
            accuracy = self.discriminant_decode(discriminant_index, x_data, y_data)
            accuracy_list.append(accuracy)
        a = 1.0 * np.array(accuracy_list)
        n = len(a)
        se = scipy.stats.sem(a)
        self.variation_strength = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        if self.verbose == 2: print("Variation_strength: ", self.variation_strength)
        return self.variation_strength
    
            
            
######################################################################################################
#Based on the chosen input, preprocessing methods and discrinant value, virtulize the process of
#parameter optimization in a model. Pass variation strength value from the variation strength function.
######################################################################################################
    def map_variation(self, prepro_index, discriminant_index, input_index):
        if self.verbose == 2: print("Creating map...")
        self.x_data, self.y_data = Combinations.input_decode(self, input_index)
        print("Preprocessing input data matrix")
        self.x_data, self.y_data = Combinations.prepro_decode(self, prepro_index, self.x_data, self.y_data)
        if not isinstance(self.x_data, pd.DataFrame):
            self.x_data = pd.DataFrame(self.x_data)
        self.variation_strength = self.Variation_strength(self.x_data, self.y_data, discriminant_index) #obtain variation_strength
        return self.variation_strength
    
    
    
######################################################################################################
#Mark unused dimension name as -1 in locations
######################################################################################################
    def remove_unused_dim(self, location, tuning_code):
        print(location)
        dim_name = list(tuning_code)
        for name in dim_name:
            if tuning_code[name] == -1:
                location[name] = -1
        print("after", location)
        return location 
    
######################################################################################################
#distribute climbers in random locations. Return a list of the status of those climbers
######################################################################################################
    def randomly_release_machines(self, prepro_index, discriminant_index, input_index, num_group, group_size):
        if self.verbose == 2: print("Initializaing machines into ", num_group, " locations. Each location has ", group_size, " climbers")
        num_machines = num_group * group_size
        climbing_machine_list = []
        variation_strength = self.map_variation(prepro_index, discriminant_index, input_index)
        for _ in range(num_machines):
            climbing_machine = Climbing_machine(self.x_data, self.y_data, self.default_tuning_code,
                                                self.default_location, prepro_index, discriminant_index,
                                                variation_strength)
            climbing_machine_list.append(climbing_machine)
        counter = 0
        location_list = []
        for i in range(num_group):
            same_location = climbing_machine_list[i * group_size].locate2_random_location()
            climbing_machine_list[counter].tuning_code_initialization(prepro_index, discriminant_index)
            same_location = self.remove_unused_dim (same_location, climbing_machine_list[counter].tuning_code)
            location_list.append(copy.deepcopy(same_location))
            
            
            counter += 1
            for _ in range(group_size-1):

                climbing_machine_list[counter].tuning_code_initialization(prepro_index, discriminant_index)
                counter += 1
                
                
        for i in range(num_group*group_size):
            group_index = int(i/group_size)
            climbing_machine_list[i].location = location_list[group_index]
        self.climbing_machine_list = climbing_machine_list
        
        if self.verbose == 2:
            print("Climbers are initialized on the following dimensions: ")
        counter2 = 1
        
        for param in list(climbing_machine_list[0].tuning_code):
                if climbing_machine_list[0].tuning_code[param] != -1:
                    self.dimension_name.append(param)
        if self.verbose == 2:
            for loc in location_list:
                print("location: ", counter2)
                for param in list(climbing_machine_list[0].tuning_code):
                    if climbing_machine_list[0].tuning_code[param] != -1:
                        print("  ", param, loc[param])
                counter2 += 1
        return climbing_machine_list
    
    
    
######################################################################################################
#excuting punishment to the climbers with bad performance and leave the top n_survival cimbers
######################################################################################################
    def kill_bad_machines(self, climbing_machine_list, n_survival = 3):
        height_list = []
        for climbing_machine in climbing_machine_list:
            height_list.append(climbing_machine.height)
        top = np.argpartition(height_list, -n_survival)[-n_survival:]
        new_climbing_machine_list = []
        for i in top:
            if self.verbose == 2: print(climbing_machine_list[i].location)
            new_climbing_machine_list.append(climbing_machine_list[i])
        self.climbing_machine_list = new_climbing_machine_list
        return new_climbing_machine_list
            
        
        
######################################################################################################
#create new climbers near the survived climbers from the last training period
######################################################################################################
    def born_on_good_location(self, good_climbing_machine_list, mode, prepro_index,
                              discriminant_index, input_index):
        good_location_list = []
        for climbing_machine in good_climbing_machine_list:
            good_location_list.append(copy.deepcopy(climbing_machine.location))
        if mode == "random_movement":
            group_size = 10
        elif mode == "360_detection":
            group_size = 1
        elif mode == "random_detection":
            group_size = 4
            
        variation = self.variation_strength
        new_climbing_machine_list = []
        for i in range(len(good_location_list)):
            same_location = good_location_list[i]
            for _ in range(group_size):
                climbing_machine = Climbing_machine(self.x_data, self.y_data,
                                                    self.default_tuning_code,
                                                    same_location,
                                                    prepro_index, discriminant_index,
                 variation)
                climbing_machine.tuning_code_initialization(prepro_index, discriminant_index)
                new_climbing_machine_list.append(climbing_machine)
        self.climbing_machine_list = new_climbing_machine_list
        return new_climbing_machine_list
    
    
######################################################################################################
#record the historical height of climbers
######################################################################################################
    def backup_historical_height(self):
        for climbing_machine in self.climbing_machine_list:
            self.accuracy_list.append(climbing_machine.height)


######################################################################################################
#plot the historical height of climbers
######################################################################################################
    def plot_historical_height(self, last_climbing_machine):
        partial_history = []
        gap = int(len(self.accuracy_list)/1000)
        if (gap == 0):
            gap = 1
        for i in range(len(self.accuracy_list)):
            if i%gap == 0:
                partial_history.append(self.accuracy_list[i])
        
        plt.plot(partial_history)
        last_climbing_machine.location = self.remove_unused_dim (last_climbing_machine.location, last_climbing_machine.tuning_code)
        plt.savefig(self.make_file_name(last_climbing_machine.location)+'.jpeg')
        plt.show()
    

######################################################################################################
#call functions to measure the height of climbers and record to the height history
######################################################################################################
    def record_height(self):
        if self.verbose == 2: print("Recording_height...")
        for i in range(len(self.climbing_machine_list)):
            self.climbing_machine_list[i].measure_height()
        self.backup_historical_height()
            
        
        
######################################################################################################
#excute one turn of training process by punishing and rewarding climbers
######################################################################################################
    def reproduce_machines(self, mode, prepro_index, discriminant_index, input_index, 
                           step_per_period, train_period = 20, n_survival = 3):
        #kill, born and move forward
        if self.verbose == 2: print("Kept top ", n_survival, " climbers and killed the remaining")
        self.climbing_machine_list = self.kill_bad_machines(self.climbing_machine_list, n_survival)
        counter = 1
        if self.verbose == 2: print("Left the following good locations")
        for climbers in self.climbing_machine_list:
            if self.verbose == 2: print("Location ", counter)
            for param in self.dimension_name:
               if self.verbose == 2: print(param, climbers.location[param])
            counter += 1
        if self.verbose == 2: print("born new climbers on good locations")
        self.climbing_machine_list = self.born_on_good_location(self.climbing_machine_list,
                                                                mode, prepro_index,
                              discriminant_index, input_index)
        
        if self.verbose == 2: print("After one movement period, browsing the height of all climbing machines...")
        new_climbing_machine_list = []
        height_list = []
        location_list = []
        for climber in self.climbing_machine_list:
            for _ in range(step_per_period):
                result = climber.move_one_step(mode, climber.height-climber.last_height)
            height_list.append(result[0])
            location_list.append(copy.deepcopy(result[1]))
            new_climbing_machine_list.append(climber)
            self.last_height_list.append(climber.height)
            self.height_list.append(climber.height)
        self.climbing_machine_list = new_climbing_machine_list
        for i in range(len(self.climbing_machine_list)):
            self.climbing_machine_list[i].height, self.climbing_machine_list[i].location = height_list[i], location_list[i]
        
       
        
        
        return self.climbing_machine_list
    
    
######################################################################################################
#create file name for the optimized model by the location of the best climber
######################################################################################################
    def make_file_name(self, dic):
        dic_name = list(dic)
        string = 'A'
        for name in dic_name:
            string = string + str(dic[name]) + 'A'
        return string
    
    
######################################################################################################
#optimize a model through n_train_period training period. The climbers would move step_per_period step
#during each period
######################################################################################################
    def run(self, mode, prepro_index, discriminant_index, input_index,
                    n_train_period = 3, train_period = 20, step_per_period = 3,
                    verbose = 0):
        self.verbose = verbose
        if mode == "random_movement":
            num_group = 10
            group_size = 10
        elif mode == "360_detection":
            if self.verbose == 2: print("Mode 360 detection: testing gradients in all directions")
            num_group = 10
            group_size = 1
        elif mode == "random_detection":
            num_group = 10
            group_size = 4
        self.climbing_machine_list = self.randomly_release_machines(prepro_index, discriminant_index, input_index, num_group, group_size)
        self.x_data = pd.DataFrame(self.x_data)
        self.record_height()
        
        if self.verbose == 2: print("After one train period, browsing the height of all climbing machines...")
        self.last_height_list = []
        self.height_list = []
        new_climbing_machine_list = []
        height_list = []
        location_list = []
        for climber in self.climbing_machine_list:
            for _ in range(step_per_period):
                result = climber.move_one_step(mode, 0)
            height_list.append(result[0])
            location_list.append(copy.deepcopy(result[1]))
            new_climbing_machine_list.append(climber)
            self.last_height_list.append(climber.height)
            self.height_list.append(climber.height)
        self.climbing_machine_list = new_climbing_machine_list
        for i in range(len(self.climbing_machine_list)):
            self.climbing_machine_list[i].height, self.climbing_machine_list[i].location = height_list[i], location_list[i]
#        for climbers in self.climbing_machine_list:
#            for param in self.dimension_name:
#                print(param, climbers.location[param])
        self.record_height()
        
        
        for i in range(n_train_period):
            if self.verbose >= 1: print("N_Train period: ", i+1)
            self.reproduce_machines(mode, prepro_index, discriminant_index,
                                    input_index, train_period, step_per_period)
            self.record_height()
        last_climbing_machine = self.kill_bad_machines(self.climbing_machine_list, 1)[0]
        self.plot_historical_height(last_climbing_machine)
        print("*********THE HEIGHT OF THE LAST SURVIVAL: ", last_climbing_machine.height)
        return last_climbing_machine
        

######################################################################################################
#optimize models based on each combination of input, preprocessing methods, and discriminant methods
#return a list of the best results from all optimized models
######################################################################################################
    def run_all_combinations(self, mode, n_train_period = 3, train_period = 20,
                             step_per_period = 3, verbose = 2):
        self.verbose = verbose
        self.set_combinations()
        input_combos = self.input_combos
        prepro_combos = self.prepro_combos
        dis_code = np.arange(1, self.dis_types + 1, 1)
        self.best_climbers_list = []
        counter = 0
        order = 12
        for inputs in input_combos:
            for prepro in prepro_combos:
                for dis in dis_code:
                    counter += 1
                    if counter >= order:
                        best_climber = self.run(mode, prepro, dis, inputs,
                                  n_train_period, train_period, step_per_period,
                                  verbose)
                        self.best_climbers_list.append(best_climber)
        return self.best_climbers_list
    
    
    
    
    
    
    
    







