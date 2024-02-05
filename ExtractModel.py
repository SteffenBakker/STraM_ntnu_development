# -*- coding: utf-8 -*-
'''
Created on Thu Jan 11 13:15:20 2024

@author: sbeike
'''

import sys
import cloudpickle

class ModelExtractor():
    
    def __init__(self, model, path):
        self.model = model
        self.path = path
        
        self.extract(self.path)
        
    def extract(self, path):
        
        with open(r"Data//solved_model//" + path +'.pickle', 'wb') as output_file: 
            print("Dumping model in pickle file.....", end="")
            cloudpickle.dump(self.model, output_file)
            print("done.")
            
        sys.stdout.flush()    