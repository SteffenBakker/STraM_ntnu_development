"""This is a toy model to test our implementation of risk aversion"""

# set wd
import os
os.chdir("\\\\home.ansatt.ntnu.no\\egbertrv\\Documents\\GitHub\\AIM_Norwegian_Freight_Model\\test_risk")

# 0. IMPORTS
import pyomo.environ as pyo
import numpy as np
#import mpisppy.utils.sputils as sputils
#from mpisppy.opt.ef import ExtensiveForm
#import mpisppy.scenario_tree as scenario_tree

# custom:
from data import Data
from model import SimpleRecourseModel
from scenario_settings import scenario_creator, scenario_denouement, option_settings_ef


# 1. RUN
base_data = Data()

scenario_names = {"scen_0", "scen_1", "scen_2", "scen_3"}

scenario_creator_kwargs = {'base_data': base_data}
ef = sputils.create_EF(
    scenario_names,
    scenario_creator,
    scenario_creator_kwargs = scenario_creator_kwargs
)

solver = pyo.SolverFactory('gurobi')
solver.options['MIPGap'] = 0.005
results = solver.solve(ef, tee = True)

#print(results)


