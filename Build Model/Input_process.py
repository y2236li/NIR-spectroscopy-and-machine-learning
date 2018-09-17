from To_bins import Bins
import sys


class Data_operation():
    def __init__ (self, data):
        self.bins = Bins(data, 10) #the default number of bin is 10
        self.data = data
        
    def find_bin(self, x):
        counter = 0
        for Onebin in self.bins.bins:
            if (x>=Onebin.lower_bound and x <=Onebin.up_bound):
                    return counter
            else:
                counter +=1
        sys.exit("Error: the value is not in the scope of the current bins")
        
    def to_binary(self):
        bi_data = []
        for i_data in range(len(self.data)):
            tmp = []
            for _ in range(self.bins.nbins):
                tmp.append(0)
            which_bin = self.find_bin(self.data[i_data])
            tmp[which_bin] += 1
            
            bi_data.append(tmp)
        self.bi_data = bi_data
        return self.bi_data
            

#data_operation = Data_operation(y_data)
#y_data_binary = data_operation.to_binary()
#print(y_data_binary)