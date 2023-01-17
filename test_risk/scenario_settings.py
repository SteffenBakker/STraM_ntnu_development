"""Scenario creator and SP settings"""

import mpisppy.utils.sputils as sputils
import mpisppy.scenario_tree as scenario_tree
from model import SimpleRecourseModel

# define scenario creator
def scenario_creator(scenario_name, **kwargs):
    scen_name_to_nr = {"scen_0": 0, "scen_1": 1, "scen_2": 2, "scen_3": 3} 
    scen_nr = scen_name_to_nr[scenario_name]

    # read data
    data = kwargs.get('base_data')

    # update scenario-dependent parameters
    data.update_scenario_dependent_parameters(scenario_name)

    # create scenario model
    model_instance = SimpleRecourseModel(data=data)    # create new instance
    model_instance.construct_model()                        # construct the model (i.e., define all constraints etc)
    model = model_instance.model                            # extract the model object

    # define root node (first stage)
    sputils.attach_root_node(model,                                     # the model
                             
                             # first-stage objective:
                             #data.c * model.x,                             # risk-neutral
                             #data.c * model.x + model.CvarAux,             # CVaR
                             data.c * model.x + data.labda * model.CvarAux, # mean-CVaR
                             
                             # list of first-stage variables
                             #[model.x]                                     # risk-neutral
                             [model.x, model.CvarAux]                       # CVaR, mean-CVaR
                             )
    
    return model

def scenario_denouement(rank, scenario_name, scenario):
    pass

def option_settings_ef():
    options = {}
    options["solvername"] = "gurobi"
    options["solver_options"] = {'MIPGap':MIPGAP}

    return options
