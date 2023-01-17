class Data:

    def __init__(self):
        self.c = 1          # first-stage cost
        self.q = 2          # second-stage cost
        self.h = 10          # demand (random!)

        # risk measure data:
        self.labda = 0.2    # risk measure lambda
        self.beta = 0.9     # risk measure beta

    
    def update_scenario_dependent_parameters(self, scenario_name):        
        
        scen_name_to_nr = {"scen_0": 0, "scen_1": 1, "scen_2": 2, "scen_3": 3}  	# scenario names to numbers
        scen_nr = scen_name_to_nr[scenario_name]
        h_scen = [10, 11, 12, 20]   # scenarios for h
        self.h = h_scen[scen_nr]


