"""This file contains the mathematical model"""

# Imports

import pyomo.opt   # we need SolverFactory,SolverStatus,TerminationCondition
import pyomo.opt.base as mobase
from pyomo.environ import *
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
import logging
import numpy as np


"""We create a simple recourse example"""

class SimpleRecourseModel:

    def __init__(self, data):

        # Import data
        self.data = data

        # Create model instance
        self.model = ConcreteModel()

        # Set solver
        self.opt = pyomo.opt.SolverFactory('gurobi')
    
    def construct_model(self):
            
        # Create variables

        # base variables
        self.model.x = Var(within=NonNegativeReals)  # first-stage production
        self.model.y = Var(within=NonNegativeReals)  # second-stage production

        # CVar variables
        self.model.CvarAux = Var(within = Reals)                # CVaR auxiliary variable
        self.model.CvarPosPart = Var(within = NonNegativeReals) # CVaR positive part

        # objective variables
        self.model.ScenObjValue = Var(within=Reals)


        # Create constraints

        # demand constraint
        def DemandRule(model):
            return self.model.x + self.model.y >= self.data.h
        self.model.DemandConstraint = Constraint(rule=DemandRule)

        # CVaR constraints: 
        
        # lower bound (for numerical reasons)
        def CvarLBRule(model):
            return self.model.CvarAux >= -1000
        self.model.CvarLBConstraint = Constraint(rule=CvarLBRule)

        # positive part constraint
        def CvarPosPartRule(model):
            return self.model.CvarPosPart >= self.model.ScenObjValue - self.model.CvarAux
        self.model.CvarPosPartConstraint = Constraint(rule=CvarPosPartRule)

                
        #Create objective

        # define scenario objective value (2nd stage)
        def ScenObjValueRule(model):
            return self.model.ScenObjValue == self.data.q * self.model.y
        self.model.ScenObjValueConstr = Constraint(rule = ScenObjValueRule)

        # risk-neutral objective function (1st + 2nd stage)
        def obj_fun_risk_neutral(model):
            return self.data.c * self.model.x + self.model.ScenObjValue

        # CVaR objective function
        def obj_fun_CVaR(model):
            return self.data.c * self.model.x + self.model.CvarAux + 1.0/(1.0 - self.data.beta) * self.model.CvarPosPart

        # risk-averse objective function
        def obj_fun_mean_CVaR(model): 
            return self.data.c * self.model.x + self.data.labda * self.model.CvarAux + (1.0 - self.data.labda) * self.model.ScenObjValue + self.data.labda / (1.0 - self.data.beta) * self.model.CvarPosPart
            #      ##### first-stage x ######   ######### first-stage u ############   ################ second-stage mean ##############   ####################### second-stage CVaR #######################

        # give objective to model
        #self.model.Obj = Objective(rule=obj_fun_risk_neutral, sense=minimize)
        #self.model.Obj = Objective(rule=obj_fun_CVaR, sense=minimize)
        self.model.Obj = Objective(rule=obj_fun_mean_CVaR, sense=minimize)

        return self.model