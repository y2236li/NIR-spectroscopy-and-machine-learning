import numpy as np
import sys


class OneBin():
    def __init__ (self, up_bound, lower_bound, n_sample, data):
        self.up_bound = up_bound
        self.lower_bound = lower_bound
        self.n_sample = n_sample
        self.data = data
        
    def info(self):
        print("Up_bound: ", self.up_bound)
        print("Lower_bound: ", self.lower_bound)
        print("Num_sample: ", self.n_sample)
        print("Data in the bin: ", self.data)
        print("\n")
        
class Bins():
    def __init__ (self, data, nbins):
        self.data = data
        self.nbins = nbins
        self.ordered = False
        self.bins = []
        self.build()
        
    def uniform_bin(self, y_data, nbins):
        y_data = [y for x in y_data for y in x]
        y_data.sort()
        self.data = y_data
        self.ordered = True
        bins_nsample = []
        bin_size = len(y_data)/nbins #the number of data in each bin
        for _ in range(nbins):
            bins_nsample.append(0)
        for i in range(len(y_data)):
            bins_nsample[np.int(i/bin_size)] += 1
        return bins_nsample
    
    def build(self):
        bins_nsample = self.uniform_bin(self.data, self.nbins)
        for i_bins_sample in range(len(bins_nsample)):
            n_sample_before = 0
            if (i_bins_sample > 0):
                n_sample_before = np.sum(bins_nsample[: (i_bins_sample)])
            onebin_data = self.data[n_sample_before:
                (n_sample_before+bins_nsample[i_bins_sample])]
            new_bin = OneBin(max(onebin_data), min(onebin_data), bins_nsample[i_bins_sample], onebin_data)
            self.bins.append(new_bin)
    
    def info(self, bin_list = []):
        print("Using data: ", self.data)
        print("The data is ordered: ", self.ordered)
        print("Num_bins: ", self.nbins)
        if (bin_list):
            if (max(bin_list)>(self.nbins-1)):
                sys.exit('Error: the bin you want to see is larger than the total number of the bins!')
            for a_bin in bin_list:
                print("bin: ", a_bin)
                self.bins[a_bin].info()
                

            
#bins = Bins(y_data, 9)
#bins.info(list(np.arange(9)))

    





