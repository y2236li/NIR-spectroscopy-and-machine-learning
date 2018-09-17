import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline




######################################################################################################
#A tool box for grabing input from excel files. Please refer to the excels in the data matrix file
#to regulate the syntax of the excel
######################################################################################################

class Grab_input():
        
    def split_data(self, x_data, y_data):
        train = np.random.randint(len(x_data), size = int(len(x_data)*0.7))
        x_train = []
        y_train = []
        x_test = []
        y_test = [] 
        for i in range(len(x_data)):
            if i in train:
                x_train.append(x_data[i])
                y_train.append(y_data[i])
            else:
                x_test.append(x_data[i])
                y_test.append(y_data[i])
        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    
    
    def extract_x_y(self, file):
        xl = pd.ExcelFile(file)
        df = xl.parse('Sheet1') #create a dataframe
        brix = df[df.columns[0]]
        df = df.drop(df.columns[0], axis = 1, inplace = False)
        colnames = df.columns.values
        x_data = []
        y_data = []
        for i in range(df.shape[0]):
            new_xdata = list(df.loc[i])
            x_data.append(new_xdata)
            new_ydata = [brix[i]]
            y_data.append(new_ydata)
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return x_data, y_data, colnames
    
    
class Inputs(Grab_input):
    ## To use this class, you should have files "Referenced Ultimate Oranges Matrix.xlsx"
    ## and "Unreferenced Ultimate Oranges Matrix.xlsx"
    
    def __init__(self):
        self.referenced_file = "Referenced Ultimate Oranges Matrix.xlsx"
        self.unreferenced_file = "Unreferenced Ultimate Oranges Matrix.xlsx"
    
    def referenced (self):
        return self.extract_x_y(self.referenced_file)
            
    def unreferenced(self):
        return self.extract_x_y(self.unreferenced_file)
    
    def derivative(self, order, referenced = True):
        if (referenced == True):
            x_data, y_data, self.colnames = self.referenced()
        else:
            x_data, y_data, self.colnames = self.unreferenced()
        
        colnames = self.colnames
        x_data = pd.DataFrame(x_data)
        new_x_data = []
        for i in range(len(x_data.index)):
            x = x_data.loc[i]
            wav_range = np.linspace(colnames[0],colnames[-1], len(colnames))
            spl = UnivariateSpline(colnames, x, s=0, k=4)
            spl_1d = spl.derivative(n=order)
            derivative = spl_1d(wav_range)
            new_x_data.append(derivative)
        return new_x_data, y_data
        
#

######################################################################################################
#The following code are used to test the module
######################################################################################################


#inputs = Inputs()
#x_data, y_data = inputs.derivative(2, False)
#x_data = pd.DataFrame(x_data)
#y_data = pd.DataFrame(y_data)
#result = pd.concat([y_data, x_data], axis=1)
#
## Create a Pandas dataframe from the data.
#df = result
#
## Create a Pandas Excel writer using XlsxWriter as the engine.
#writer = pd.ExcelWriter('Second Derivative.xlsx', engine='xlsxwriter')
#
## Convert the dataframe to an XlsxWriter Excel object.
#df.to_excel(writer, sheet_name='Sheet1')
#
## Close the Pandas Excel writer and output the Excel file.
#writer.save()
#        
#        
#        
        
        
        
        
        
        
        
        
        
        
        
