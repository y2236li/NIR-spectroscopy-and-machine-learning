import numpy as np

######################################################################################################
#After find all combinations, the module filter and leave qualified combinations
######################################################################################################
class All_combos():
    def __init__(self, num):
        self.num = num
        self.combo = []
        
    def check_format(self, strNum):
        num = []
        for i in strNum:
            num.append(np.int(i))
        for i in num:
            if i > self.num - 1:
                return False
        valid = True
        zero_begin = True
        zero_end = False
        zero_begin_counter = 0
        for i in range(len(num)):
            if i < len(num)-1:
                if (zero_begin == True):
                    if num[i] == 0:
                        zero_begin_counter += 1
                    else:
                        zero_begin = False
                if(zero_begin == False and zero_end == False):
                    if (num[i] >= num[i+1] and (num[i] != 0 and num[i+1] != 0)):
                        valid = False
                    elif (num[i] == 0 or num[i+1] == 0):
                        zero_end = True
                        
                elif(zero_end == True):
                    if (num[i] != 0 or num[i+1] != 0):
                        valid = False
            if (i == len(num)-1 and valid == True):
                if (zero_begin == False and zero_end == False):
                    if (num[i] < num[i-1] and num[i]!=0):
                        valid = False
                if (zero_end == True):
                    valid = True
#        if (zero_begin == True and zero_end == False):
#            if zero_begin_counter > 1:
#                valid = False
        return valid
    
    def all_format(self):
        candidates = np.arange(10**self.num)
        candidates_str = []
        for c in candidates:
            candidates_str.append(str(c))
        for i in range(len(candidates_str)):
            candidates_str[i] = str(candidates_str[i])
            zeros = (self.num-len(candidates_str[i]))*'0'
            candidates_str[i] = zeros + candidates_str[i]
        return candidates_str
    
    def remove_zero_end(self, strNum):
        ## use this function after check_format
        zero_begin = True
        num = []
        for i in strNum:
            num.append(np.int(i))
        for i in range(len(num)):
            if num[i] != 0 and zero_begin == True:
                zero_begin = False
            elif zero_begin == False:
                if num[i] == 0:
                    return strNum[:i]
        if zero_begin == True:
            return '0'
        elif zero_begin == False:
            return strNum
        
    def remove_serialZero_begin(self, strNum):
        ## use this function after remove_serialZero_begin
        begin_by_zero = False
        num = []
        for i in strNum:
            num.append(np.int(i))
        for i in range(len(num)):
            if begin_by_zero == False and num[i] == 0:
                begin_by_zero = True
            elif begin_by_zero == True:
                if num[i] != 0:
                    return '0' + strNum[i:]
        if begin_by_zero == True:
            return '0'
        return strNum
            
    
    def index_result(self):
        all_format = self.all_format()
        TF_result = []
        final_result = []
        for i in all_format:
            TF_result.append(self.check_format(i))
        for i in range(len(all_format)):
            if TF_result[i]:
                all_format[i] = self.remove_zero_end(all_format[i])
                all_format[i] = self.remove_serialZero_begin(all_format[i])
                final_result.append((all_format[i]))
        return final_result
        
        
#all_combos = All_combos(3)
#print(all_combos.index_result())

    
    
    
    
    
    

                        
