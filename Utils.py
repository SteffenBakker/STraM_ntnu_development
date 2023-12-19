import sys
import os

class Logger(object):
    def __init__(self,filename='', log_to_file=False):
        self.terminal = sys.stdout
        self.log_to_file = log_to_file
        if log_to_file:
            if not os.path.exists("Data/Output"):
                os.makedirs("Data/Output")
            self.log = open(r"Data/output/"+filename+"_logfile.log", "w")
   
    def write(self, message):
        self.terminal.write(message)
        if self.log_to_file:
            self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        #pass 
        self.terminal.flush()
        if self.log_to_file:
            self.log.flush()   

