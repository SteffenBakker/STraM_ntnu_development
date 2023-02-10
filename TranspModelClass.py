

#Pyomo
import pyomo.opt   # we need SolverFactory,SolverStatus,TerminationCondition
import pyomo.opt.base as mobase
from pyomo.environ import *
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints

import logging
from Data.settings import *
from Data.Create_Sets_Class import TransportSets   #FreightTransportModel.Data.Create_Sets_Class
from ExtractModelResults import OutputData
import time

import os
import numpy as np

############# Class ################

class RiskInformation:

    def __init__(self, cvar_coeff, cvar_alpha):
        self.cvar_coeff = cvar_coeff    # \lambda: coefficient for CVaR in the mean-cvar objective
        self.cvar_alpha = cvar_alpha    # \alpha:  indicates how far in the tail we care about risk (0.9 means we care about worst 10%)

class TranspModel:

    def __init__(self, data, risk_info):

        self.results = None  # results is a structure filled out later in solve_model()
        self.status = None  # status is a string filled out later in solve_model()
        self.model = ConcreteModel()
        self.opt = pyomo.opt.SolverFactory('gurobi') #gurobi
        self.opt.options['FeasibilityTol'] = 10**(-3) #the standard of 10**(-6) gives a constraint violation warning
        self.opt.options['MIPGap']= MIPGAP # 'TimeLimit':600 (seconds)
        self.data = data
        self.risk_info = risk_info # stores parameters of the risk measure (i.e., lambda and alpha for mean-CVaR)

        self.calculate_max_transp_amount_exact = False

        self.solve_base_year = False
        self.last_time_period = False

        self.NoBalancingTrips = False

    def construct_model(self):
        
        #Significant speed improvements can be obtained using the LinearExpression object when there are long, dense, linear expressions.
        # USe linearexpressions: https://pyomo.readthedocs.io/en/stable/advanced_topics/linearexpression.html
        

        "VARIABLES"
        # Binary, NonNegativeReals, PositiveReals, etc
        self.model.x_flow = Var(self.data.AFPT_S, within=NonNegativeReals)
        self.model.b_flow = Var(self.data.AFVT_S, within=NonNegativeReals)
        self.model.h_path = Var(self.data.KPT_S, within=NonNegativeReals)# flow on paths K,p
        self.model.h_path_balancing = Var(self.data.KVT_S, within=NonNegativeReals,initialize=1)# flow on paths K,p
        if self.NoBalancingTrips:
            for (i,j,m,r,f,v,t,s) in self.data.AFVT_S:
                a = (i,j,m,r)
                self.model.b_flow[(a,f,v,t,s)].fix(0)
            for (k,v,t) in self.data.KVT_S:
                self.model.h_path_balancing[(k,v,t,s)].fix(0)
        self.model.StageCosts = Var(self.data.T_TIME_PERIODS_S, within = NonNegativeReals)
        
        self.model.epsilon_edge = Var(self.data.ET_RAIL_S, within = Binary) #within = Binary
        self.model.upsilon_upg = Var(self.data.UT_UPG_S, within = Binary) #bin.variable for investments upgrade/new infrastructure u at link l, time period t
        self.model.nu_node = Var(self.data.NCMT_S, within = Binary) #step-wise investment in terminals
        self.model.y_charge = Var(self.data.EFT_CHARGE_S, within=NonNegativeReals)

        self.model.z_emission = Var(self.data.TS_S, within = NonNegativeReals)

        self.model.total_emissions = Var(self.data.TS_S, within=NonNegativeReals) #instead of T_PERIODS!
                                        # bounds=emission_bound)  # just a variable to help with output
        
        self.model.q_transp_amount = Var(self.data.MFT_S, within=NonNegativeReals)
        self.model.q_max_transp_amount = Var(self.data.MF_S, within=NonNegativeReals)

        # auxiliary transport amount for tracking diffusion paths:
        self.model.q_aux_transp_amount = Var(self.data.MFT_NEW_YEARLY_S, within = NonNegativeReals) 
        # total transport amount per mode
        self.model.q_mode_total_transp_amount = Var(self.data.MT_S, within = NonNegativeReals) #note: only measured in decision years
        
        MAX_TRANSP_PENALTY = MAX_TRANSPORT_AMOUNT_PENALTY
        if self.calculate_max_transp_amount_exact:
            self.model.chi_max_transp = Var(self.data.MFT_S, within = Binary)
            MAX_TRANSP_PENALTY = 0

        #COST_VARIABLES

        self.model.TranspOpexCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def TranspOpexCost(model, t,s):
            return (self.model.TranspOpexCost[t,s] == sum((self.data.C_TRANSP_COST[(i,j,m,r,f,p,t,s)])*self.model.x_flow[(i,j,m,r,f,p,t,s)] 
                                      for p in self.data.P_PRODUCTS for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m]) )
        self.model.TranspOpexCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=TranspOpexCost)
        
        self.model.TranspCO2Cost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def TranspCO2Cost(model, t,s):
            return (self.model.TranspCO2Cost[t,s] == sum((self.data.C_CO2[(i,j,m,r,f,p,t)])*self.model.x_flow[(i,j,m,r,f,p,t,s)] 
                                      for p in self.data.P_PRODUCTS for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m]) )
        self.model.TranspCO2CostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=TranspCO2Cost)
        
        self.model.TranspOpexCostB = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def TranspOpexCostB(model, t,s):
            return (self.model.TranspOpexCostB[t,s] == sum( EMPTY_VEHICLE_FACTOR*(self.data.C_TRANSP_COST[(i,j,m,r,f,self.data.cheapest_product_per_vehicle[(m,f,t,v)],t,s)]) * self.model.b_flow[(i,j,m,r,f,v,t,s)] 
                                    for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m] )  )
        self.model.TranspOpexCostBConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=TranspOpexCostB)
        
        self.model.TranspCO2CostB = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def TranspCO2CostB(model, t,s):
            return (self.model.TranspCO2CostB[t,s] == sum( EMPTY_VEHICLE_FACTOR*(self.data.C_CO2[(i,j,m,r,f,self.data.cheapest_product_per_vehicle[(m,f,t,v)],t)]) * self.model.b_flow[(i,j,m,r,f,v,t,s)] 
                                    for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m] )  )
        self.model.TranspCO2CostBConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=TranspCO2CostB)
        
        self.model.TransfCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def TransfCost(model, t,s):
            return (self.model.TransfCost[t,s] == sum(self.data.C_TRANSFER[(k,p)]*self.model.h_path[k,p,t,s] for p in self.data.P_PRODUCTS  for k in self.data.MULTI_MODE_PATHS) )
        self.model.TransfCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=TransfCost)
        
        self.model.EdgeCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def EdgeCost(model, t,s):
            return (self.model.EdgeCost[t,s] == sum(self.data.C_EDGE_RAIL[e]*self.model.epsilon_edge[(e,t,s)] for e in self.data.E_EDGES_RAIL) )
        self.model.EdgeCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=EdgeCost)
        
        self.model.NodeCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def NodeCost(model, t,s):
            return (self.model.NodeCost[t,s] == sum(self.data.C_NODE[(i,c,m)]*self.model.nu_node[(i,c,m,t,s)] for (i, m) in self.data.NM_LIST_CAP for c in self.data.TERMINAL_TYPE[m]) )
        self.model.NodeCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=NodeCost)
        
        self.model.UpgCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def UpgCost(model, t,s):
            return (self.model.UpgCost[t,s] == sum(self.data.C_UPG[(e,f)]*self.model.upsilon_upg[(e,f,t,s)] for (e,f) in self.data.U_UPGRADE))
        self.model.UpgCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=UpgCost)
        
        self.model.ChargeCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def ChargeCost(model, t,s):
            return (self.model.ChargeCost[t,s] == sum(self.data.C_CHARGE[(e,f)]*self.model.y_charge[(e,f,t,s)] for (e,f) in self.data.EF_CHARGING) )
        self.model.ChargeCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=ChargeCost)
        
        self.model.MaxTranspPenaltyCost = Var(self.data.S_SCENARIOS, within=NonNegativeReals)
        def MaxTranspPenaltyCost(model,s): #TEST
            return (self.model.MaxTranspPenaltyCost[s] == MAX_TRANSP_PENALTY*sum(self.model.q_max_transp_amount[m,f,s] for m in self.data.M_MODES for f in self.data.FM_FUEL[m]) )
        self.model.MaxTranspPenaltyCostConstr = Constraint(self.data.S_SCENARIOS, rule=MaxTranspPenaltyCost)

        # CVaR variables
        
        # CVaR auxiliary variable (corresponds to u)
        self.model.CvarAux = Var(within=Reals)

        #CVaR positive part variable (corresponds to z = (f - u)^+)
        self.model.CvarPosPart = Var(self.data.S_SCENARIOS, within=NonNegativeReals)               #note: includes one positive part constraint: z \geq 0


        "OBJECTIVE"
        #-------------------------

        def StageCostsVar(model, t,s):  #TODO: QUESTION: WHY DO WE USE "model" as input instead of "self"? (not just here, but everywhere)

            # SOME QUICK TESTING SHOWED THAT SUM_PRODUCT IS QUITE A BIT SLOWER THAN SIMPLY TAKING THE SUM...
            yearly_transp_cost = (self.model.TranspOpexCost[t,s] + self.model.TranspCO2Cost[t,s] + self.model.TranspOpexCostB[t,s] + self.model.TranspCO2CostB[t,s]) 
            
            #sum(base_data.D_DISCOUNT_RATE**n for n in list(range(10)))
            if t == self.data.T_TIME_PERIODS[-1]:
                factor = self.data.D_DISCOUNT_RATE**self.data.Y_YEARS[t][0] * (1/(1-self.data.D_DISCOUNT_RATE))      #was 2.77, becomes 9
            else:
                factor = sum(self.data.D_DISCOUNT_RATE**n for n in self.data.Y_YEARS[t])
            opex_costs = factor*(yearly_transp_cost+self.model.TransfCost[t,s]) 
            
            delta = self.data.D_DISCOUNT_RATE**self.data.Y_YEARS[t][0]
            investment_costs = self.model.EdgeCost[t,s] + self.model.NodeCost[t,s] + self.model.UpgCost[t,s] + self.model.ChargeCost[t,s] 
            
            return (self.model.StageCosts[t,s] == (opex_costs + delta*investment_costs + EMISSION_VIOLATION_PENALTY*self.model.z_emission[(t,s)]))
        self.model.stage_costs = Constraint(self.data.T_TIME_PERIODS_S, rule = StageCostsVar)
        
     
        # first-stage objective value variable
        self.model.FirstStageCosts = Var(self.data.S_SCENARIOS, within = Reals)
        def FirstStageCostsRule(model,s):
            return self.model.FirstStageCosts[s] == sum(self.model.StageCosts[t,s] for t in self.data.T_TIME_PERIODS if t in self.data.T_TIME_FIRST_STAGE)
        self.model.FirstStageCostsConstr = Constraint(self.data.S_SCENARIOS, rule = FirstStageCostsRule)

        # second-stage objective value variable
        self.model.SecondStageCosts = Var(self.data.S_SCENARIOS, within = Reals)
        def SecondStageCostsRule(model,s):
            return self.model.SecondStageCosts[s] == sum(self.model.StageCosts[t,s] for t in self.data.T_TIME_PERIODS if t in self.data.T_TIME_SECOND_STAGE) + self.model.MaxTranspPenaltyCost[s]
        self.model.SecondStageCostsConstr = Constraint(self.data.S_SCENARIOS,rule = SecondStageCostsRule)

        # scenario objective value variable (risk-neutral model would take expectation over this in the objective)
        #self.model.ScenObjValue = Var(within=Reals)
        #def ScenObjValue(model):
        #    return self.model.ScenObjValue == sum(self.model.StageCosts[t] for t in self.data.T_TIME_PERIODS) +  self.model.MaxTranspPenaltyCost  # corresponds to f(x,\xi)
        #self.model.ScenObjValueConstr = Constraint(rule = ScenObjValue)
        
        #CVaR positive part constraint
        def CvarRule(model,s):
            return self.model.CvarPosPart[s] >= self.model.SecondStageCosts[s] - self.model.CvarAux       # z \geq v - u
        self.model.CvarPosPartConstr = Constraint(self.data.S_SCENARIOS, rule=CvarRule) # add to model

        # lower bound on auxiliary CVaR variable (needed to avoid numerical issues)
        cvar_aux_lb = -1e10 # (hardcoded). 10 seems to be more accurate than 11
        def CvarAuxLBRule(model):
            return self.model.CvarAux >= cvar_aux_lb 
        self.model.CvarAuxLBConstr = Constraint(rule=CvarAuxLBRule) # add to model

        # mean-CVaR:
        def objfun_risk_averse(model):          
            # scenario objective value for risk-averse (mean-CVaR) model:
            mean_cvar_scen_obj_value = 1/len(self.data.S_SCENARIOS)*sum(self.model.FirstStageCosts[s] + self.risk_info.cvar_coeff * self.model.CvarAux + (1 - self.risk_info.cvar_coeff) * self.model.SecondStageCosts[s] + self.risk_info.cvar_coeff / (1 - self.risk_info.cvar_alpha) * self.model.CvarPosPart[s] for s in self.data.S_SCENARIOS)
            #                          ####### c*x ##############   ############# lambda * u #####################   ####################### (1 - lambda) * q*y ##################   ############### lambda / (1 - alpha) * q*y #########################################                                                             
            return mean_cvar_scen_obj_value

        # # pure CVaR:
        # def objfun_pure_cvar(model):          
        #     # scenario objective value for risk-averse (mean-CVaR) model:
        #     cvar_scen_obj_value = self.model.FirstStageCosts + self.model.CvarAux + 1.0/(1 - self.risk_info.cvar_alpha) * self.model.CvarPosPart  
        #     return cvar_scen_obj_value

        # # risk-neutral #NOTE: TEMPORARY
        # def objfun_risk_neutral(model):          
        #     # scenario objective value for risk-neutral model:
        #     risk_neutral_scen_obj_value = self.model.FirstStageCosts + self.model.SecondStageCosts
        #     return risk_neutral_scen_obj_value

        



        # give objective function to model
        self.model.objective_function = Objective(rule=objfun_risk_averse, sense=minimize) #risk-averse
        #self.model.Obj = Objective(rule=objfun_risk_neutral, sense=minimize) #TEMPORARY: risk-neutral
        


        "CONSTRAINTS"

        # DEMAND
                
        def FlowRule(model, o, d, p, t,s):
            return sum(self.model.h_path[(k, p, t,s)] for k in self.data.OD_PATHS[(o, d)]) >= self.data.D_DEMAND[
                (o, d, p, t)]
        # THIS SHOULD BE AN EQUALITY; BUT THEN THE PROBLEM GETS EASIER WITH A LARGER THAN OR EQUAL
        self.model.Flow = Constraint(self.data.ODPTS_CONSTR_S, rule=FlowRule)
        

        # PATHFLOW

        def PathArcRule(model, i, j, m, r, p, t,s):
            a= (i,j,m,r)
            return sum(self.model.x_flow[a, f, p, t,s] for f in self.data.FM_FUEL[m]) == sum(
                self.model.h_path[k, p, t,s] for k in self.data.KA_PATHS[a] )
        self.model.PathArcRel = Constraint(self.data.APT_CONSTR_S, rule=PathArcRule)

        
        if not self.NoBalancingTrips:
            def PathArcRuleBalancing(model, i, j, m, r, v, t,s):
                a= (i,j,m,r)
                return sum(self.model.b_flow[a, f, v, t,s] for f in self.data.FM_FUEL[m]) == sum(
                    self.model.h_path_balancing[k, v, t,s] for k in self.data.KA_PATHS_UNIMODAL[a] )
            self.model.PathArcRelBalance = Constraint(self.data.AVT_CONSTR_S, rule=PathArcRuleBalancing)

            # FLEET BALANCING

            deviation = 1

            def FleetBalance1(model, n,m,f,v, t,s):
                disbalance_in_node = (sum(self.model.x_flow[(a, f, p, t,s)] for a in self.data.ANM_ARCS_IN[(n,m)] for p in self.data.PV_PRODUCTS[v]) - 
                        sum(self.model.x_flow[(a, f, p, t, s)] for a in self.data.ANM_ARCS_OUT[(n,m)] for p in self.data.PV_PRODUCTS[v]))  
                empty_trips = (sum(self.model.b_flow[(a, f, v, t,s)] for a in self.data.ANM_ARCS_OUT[(n,m)]) -
                            sum(self.model.b_flow[(a, f, v, t,s)] for a in self.data.ANM_ARCS_IN[(n,m)]))            
                return  (-deviation, disbalance_in_node - empty_trips,deviation)  
            self.model.FleetBalance1 = Constraint(self.data.NMFVT_CONSTR_S, rule=FleetBalance1)



        #-----------------------------------------------#

        # EMISSIONS
        def emissions_rule(model, t,s):
            return (
                self.model.total_emissions[t,s] == sum(
                self.data.E_EMISSIONS[i,j,m,r,f, p, t] * self.model.x_flow[i,j,m,r,f, p, t,s] for p in self.data.P_PRODUCTS
                for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m]) + 
                sum(self.data.E_EMISSIONS[i,j,m,r,f, self.data.cheapest_product_per_vehicle[(m,f,t,v)], t]*EMPTY_VEHICLE_FACTOR * self.model.b_flow[i,j,m,r,f, v, t,s] 
                    for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m])
                )
        
        self.model.Emissions = Constraint(self.data.TS_CONSTR_S, rule=emissions_rule) #removed self.data.T_TIME_PERIODS

        # Emission limit
        # IF ERROR HERE -> REMEMBER TO RUN INITIALIZATION MODEL TO DETERMINE THE ABSOLUTE EMISSION AMOUNT
        def EmissionCapRule(model, t,s):
            return self.model.total_emissions[t,s] <= self.data.EMISSION_CAP_RELATIVE[t]/100*self.data.EMISSION_CAP_ABSOLUTE_BASE_YEAR*1.01 + self.model.z_emission[t,s]
        self.model.EmissionCap = Constraint(self.data.TS_NO_BASE_YEAR_CONSTR_S, rule=EmissionCapRule)
        
        #-----------------------------------------------#
        
        #CAPACITY
        def CapacitatedFlowRule(model,i,j,m,r,ii,jj,mm,rr,t,s):
            e = (i,j,m,r)
            a = (ii,jj,mm,rr)
            return (sum(self.model.x_flow[a, f, p, t, s] for p in self.data.P_PRODUCTS for f in self.data.FM_FUEL[m]) + 
                    sum(self.model.b_flow[a, f, v, t, s] for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m] ) <= 0.5*(self.data.Q_EDGE_BASE_RAIL[e] +
                   + self.data.Q_EDGE_RAIL[e] * sum(self.model.epsilon_edge[e, tau, s] for tau in self.data.T_TIME_PERIODS if tau <= (t-self.data.LEAD_TIME_EDGE_RAIL[e]))))
        self.model.CapacitatedFlow = Constraint(self.data.EAT_RAIL_CONSTR_S, rule = CapacitatedFlowRule)
        
        #Num expansions
        def ExpansionLimitRule(model,i,j,m,r,s):
            e = (i,j,m,r)
            return (sum(self.model.epsilon_edge[(e,t,s)] for t in self.data.T_TIME_PERIODS) <= 1)
        if len(self.data.T_TIME_PERIODS)>1:
            self.model.ExpansionCap = Constraint(self.data.E_EDGES_RAIL_S, rule = ExpansionLimitRule)
        
        #Terminal capacity constraint. We keep the old notation here, so we can distinguish between OD and transfer, if they take up different capacity.
        def TerminalCapRule(model, i, c, m,t,s):
            return(sum(self.model.h_path[k, p, t,s] for k in self.data.ORIGIN_PATHS[(i,m)] for p in self.data.PT[c]) + 
                   sum(self.model.h_path[k, p, t,s] for k in self.data.DESTINATION_PATHS[(i,m)] for p in self.data.PT[c]) +
                   sum(self.model.h_path[k,p,t,s] for k in self.data.TRANSFER_PATHS[(i,m)] for p in self.data.PT[c]) <= 
                   self.data.Q_NODE_BASE[i,c,m]+self.data.Q_NODE[i,c,m]*sum(self.model.nu_node[i,c,m,tau,s] for tau in self.data.T_TIME_PERIODS if tau <= (t-self.data.LEAD_TIME_NODE[i,c,m])))
        self.model.TerminalCap = Constraint(self.data.NCMT_CONSTR_S, rule = TerminalCapRule)
        
        #Num expansions of terminal NEW -- how many times you can perform a step-wise increase of the capacity
        def TerminalCapExpRule(model, i, c,m,s):
            return(sum(self.model.nu_node[i,c,m,t,s] for t in self.data.T_TIME_PERIODS) <= 1) # THIS WAS AT 4 .... self.data.INV_NODE[i,m,c])
        if len(self.data.T_TIME_PERIODS)>1:
            self.model.TerminalCapExp = Constraint(self.data.NCM_S, rule = TerminalCapExpRule)

        #Charging / Filling
        def ChargingCapArcRule(model, i, j, m, r,f, t,s):
            e = (i, j, m, r)
            return (sum(self.model.x_flow[a,f,p, t,s] for p in self.data.P_PRODUCTS
                       for a in self.data.AE_ARCS[e]) + sum(self.model.b_flow[a,f,v, t,s] for a in self.data.AE_ARCS[e] 
                        for v in self.data.VEHICLE_TYPES_M[m]) <= self.data.Q_CHARGE_BASE[(e,f)] +
                   sum(self.model.y_charge[(e,f,tau,s)] for tau in self.data.T_TIME_PERIODS if tau <= (t-self.data.LEAD_TIME_CHARGING[(e,f)])))
        self.model.ChargingCapArc = Constraint(self.data.EFT_CHARGE_CONSTR_S, rule=ChargingCapArcRule)
        #AIM also looked into charging infrastructure in NODES

        #Upgrading
        def InvestmentInfraRule(model,i,j,m,r,f,t,s):
            e = (i,j,m,r)
            return (sum(self.model.x_flow[a,f,p,t,s] for p in self.data.P_PRODUCTS for a in self.data.AE_ARCS[e])
                    <= self.data.BIG_M_UPG[e]*sum(self.model.upsilon_upg[e,f,tau,s] for tau in self.data.T_TIME_PERIODS if tau <= (t-self.data.LEAD_TIME_UPGRADE[(e,f)])))
        self.model.InvestmentInfra = Constraint(self.data.UT_UPG_CONSTR_S, rule = InvestmentInfraRule)
    
        #-----------------------------------------------#
    
        #TransportArbeid
        def TotalTransportAmountRule(model,m,f,t,s):
            return (self.model.q_transp_amount[m,f,t,s] == sum( self.data.AVG_DISTANCE[a]*self.model.x_flow[a,f,p,t,s] for p in self.data.P_PRODUCTS 
                                                            for a in self.data.AM_ARCS[m]))
        self.model.TotalTranspAmount = Constraint(self.data.MFT_CONSTR_S, rule = TotalTransportAmountRule)

        #Technology maturity limit upper bound (which also comes from bass diffusion)
        def TechMaturityLimitRule(model, m, f, t,s):
            return (self.model.q_transp_amount[(m,f,t,s)] <= self.data.R_TECH_READINESS_MATURITY[(m,f,t,s)]/100*sum(self.model.q_transp_amount[(m,ff,t,s)] for ff in self.data.FM_FUEL[m]))   #TO DO: CHANGE THIS Q_TECH to R*M
        self.model.TechMaturityLimit = Constraint(self.data.MFT_MATURITY_CONSTR_S, rule = TechMaturityLimitRule)


        if not self.last_time_period:

            #Auxiliary transport amount (q_aux equal to q in all decision periods t)
            def AuxTransportAmountRule(model,m,f,t,s):
                return (self.model.q_aux_transp_amount[m,f,t,s] == self.model.q_transp_amount[m,f,t,s]) #auxiliary q variable equal to "normal" q variable 
            self.model.AuxTranspAmount = Constraint(self.data.MFT_NEW_S, rule = AuxTransportAmountRule) #note: only in decision periods

            #Total mode transport amount (q_mode_total equal to sum over q_aux for all f in F[m], for decision periods)
            def ModeTotalTransportAmountRule(model,m,t,s):
                return (self.model.q_mode_total_transp_amount[m,t,s] == sum( self.model.q_transp_amount[m,f,t,s] for f in self.data.FM_FUEL[m] ))
            self.model.ModeTotalTransportAmount = Constraint(self.data.MT_S, rule = ModeTotalTransportAmountRule) 

            #Initialize the transport amounts (put an upper bound at first)
            def InitTranspAmountRule(model, m, f, t,s):
                return (self.model.q_transp_amount[(m,f,t,s)] <= self.data.Q_SHARE_INIT_MAX[(m,f,t)]/100*sum(self.model.q_transp_amount[(m,ff,t,s)] for ff in self.data.FM_FUEL[m]))   #TO DO: CHANGE THIS Q_TECH to R*M
            self.model.InitTranspAmount = Constraint(self.data.MFT_INIT_TRANSP_SHARE_S, rule = InitTranspAmountRule)
            

            #Max TransportArbeid
            def MaxTransportAmountRule(model,m,f,t,s):
                return (self.model.q_max_transp_amount[m,f,s] >= self.model.q_transp_amount[m,f,t,s])
            self.model.MaxTranspAmount = Constraint(self.data.MFT_S, rule = MaxTransportAmountRule)
            
            if self.calculate_max_transp_amount_exact:  #this goes much slower!
                def MaxTransportAmountRule2(model,m,f,t,s):
                    M = np.max(list(self.data.AVG_DISTANCE.values())) * self.data.D_DEMAND_AGGR[t]  #maybe we can use mean as well
                    return (self.model.q_max_transp_amount[m,f,s] <= self.model.q_transp_amount[m,f,t,s] + M*(1-self.model.chi_max_transp[m,f,t,s]))
                self.model.MaxTranspAmount2 = Constraint(self.data.MFT_S, rule = MaxTransportAmountRule2)

                def MaxTransportAmountRule3(model,m,f,s):
                    return (sum(self.model.chi_max_transp[m,f,t,s] for t in self.data.T_TIME_PERIODS) == 1)
                self.model.MaxTranspAmount3 = Constraint(self.data.MF_S, rule = MaxTransportAmountRule3)
                
            #Fleet Renewal
            def FleetRenewalRule(model,m,f, t,s):
                decrease = self.model.q_transp_amount[(m,f,self.data.T_MIN1[t],s)] - self.model.q_transp_amount[(m,f,t,s)]
                factor = (t - self.data.T_MIN1[t]) / self.data.LIFETIME[(m,f)]
                return (decrease <= factor*self.model.q_max_transp_amount[m,f,s])
            self.model.FleetRenewal = Constraint(self.data.MFT_MIN0_S, rule = FleetRenewalRule)

            
            #-- Bass diffusion ----------------------

            #Bass diffusion paths first period (initial values): q[t] <= (2022 - t_0)^+ * alpha * q_bar[t]
            def BassDiffusionRuleFirstPeriod(model,m,f,t,s):
                pos_part = max(self.data.T_TIME_PERIODS[0] - self.data.tech_active_bass_model[(m,f,s)].t_0, 0)
                return ( self.model.q_transp_amount[m,f,t,s] <= pos_part * self.data.tech_active_bass_model[(m,f,s)].p * self.model.q_mode_total_transp_amount[m,t,s] )
            self.model.BassDiffusionFirstPeriod = Constraint(self.data.MFT_NEW_FIRST_PERIOD_S, rule = BassDiffusionRuleFirstPeriod)

            # only add remaining Bass diffusion constraints if we run the full model (not just first period in the initialization run)
            if len(self.data.T_TIME_PERIODS) > 1:
                #Bass diffusion paths (1st stage): change in q is at most alpha * q_bar[t-1] + beta * q[t-1]    (based on pessimistic beta)
                def BassDiffusionRuleFirstStage(model,m,f,t,s):
                    diff_has_started = (t >= self.data.tech_active_bass_model[(m,f,s)].t_0) # boolean indicating whether diffusion process has started at time t
                    return ( self.model.q_aux_transp_amount[m,f,t,s] - self.model.q_aux_transp_amount[m,f,t-1,s] 
                        <= diff_has_started * (self.data.tech_active_bass_model[(m,f,s)].p * self.model.q_mode_total_transp_amount[m,self.data.T_MOST_RECENT_DECISION_PERIOD[t-1],s]
                        #+ (1 - self.data.tech_scen_p_q_variation[(m,f)]) * self.data.tech_active_bass_model[(m,f,s)].q * self.model.q_aux_transp_amount[m,f,t-1] ))   #initial pessimistic path
                        + self.data.tech_active_bass_model[(m,f,s)].q * self.model.q_aux_transp_amount[m,f,t-1,s] ))   #initial base path
                self.model.BassDiffusionFirstStage = Constraint(self.data.MFT_NEW_YEARLY_FIRST_STAGE_MIN0_S, rule = BassDiffusionRuleFirstStage)

                # Bass diffusion paths (2nd stage): change in q is at most alpha * q_bar[t-1] + beta * q[t-1]   (based on scenario beta)
                def BassDiffusionRuleSecondStage(model,m,f,t,s):
                    diff_has_started = (t >= self.data.tech_active_bass_model[(m,f,s)].t_0) # boolean indicating whether diffusion process has started at time t
                    return ( self.model.q_aux_transp_amount[m,f,t,s] - self.model.q_aux_transp_amount[m,f,t-1,s] 
                        <= diff_has_started * ( self.data.tech_active_bass_model[(m,f,s)].p * self.model.q_mode_total_transp_amount[m,self.data.T_MOST_RECENT_DECISION_PERIOD[t-1],s] 
                        + self.data.tech_active_bass_model[(m,f,s)].q * self.model.q_aux_transp_amount[m,f,t-1,s] ) )  
                        # q(t) - q(t-1) <= alpha * q_bar(|_t-1_|) + beta * q(t-1)
                self.model.BassDiffusionSecondStage = Constraint(self.data.MFT_NEW_YEARLY_SECOND_STAGE_S, rule = BassDiffusionRuleSecondStage)

        #-----------------------------------------------#
        #    Specific constraints
        #-----------------------------------------------#

            #Do not use too much road
            def RoadDevelopmentRule(model, t,s):
                m = "Road"
                return (sum(self.model.q_transp_amount[(m,f,t,s)] for f in self.data.FM_FUEL[m]) <= \
                    GROWTH_ON_ROAD*sum(self.model.q_transp_amount[(m,f,self.data.T_TIME_PERIODS[0],s)] for f in self.data.FM_FUEL[m]))   
            self.model.RoadDevelopment = Constraint(self.data.T_TIME_PERIODS_S, rule = RoadDevelopmentRule)

        if self.solve_base_year:

            def InitialModeSplitLower(model,m,s):
                t0 = self.data.T_TIME_PERIODS[0]
                total_transport_amount = sum(self.model.q_transp_amount[(mm,f,t0,s)] for mm in self.data.M_MODES for f in self.data.FM_FUEL[mm])
                modal_transport_amount = sum(self.model.q_transp_amount[(m,f,t0,s)] for f in self.data.FM_FUEL[m])
                return ( INIT_MODE_SPLIT_LOWER[m]*total_transport_amount <= modal_transport_amount ) 
            self.model.InitModeSplitLower = Constraint(self.data.M_MODES_S,rule = InitialModeSplitLower)

            def InitialModeSplitUpper(model,m,s):
                t0 = self.data.T_TIME_PERIODS[0]
                total_transport_amount = sum(self.model.q_transp_amount[(mm,f,t0,s)] for mm in self.data.M_MODES for f in self.data.FM_FUEL[mm])
                modal_transport_amount = sum(self.model.q_transp_amount[(m,f,t0,s)] for f in self.data.FM_FUEL[m])
                return (modal_transport_amount <= INIT_MODE_SPLIT_UPPER[m]*total_transport_amount) 
            self.model.InitModeSplitUpper = Constraint(self.data.M_MODES_S,rule = InitialModeSplitUpper)

        #-----------------------------------------------#

        # NON ANTICIPATIVIY


        #Do not model in a smart way, just write it out!
        #(next time, model the variables in a dictionary, so self.model.vars['x_flow'][(a,f,p,t,s)])
    
        def combinations(list_of_tuples, list):
            list_of_tuples2 = []
            for tpl in list_of_tuples:
                for l in list:
                    tpl2 = tpl + (l,)
                    list_of_tuples2.append(tpl2) 
            return list_of_tuples2

        def Nonanticipativity_x(model,i,j,m,r,f,p,t,s,ss):
            a = i,j,m,r
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): # AND NOT ALREADY ADDED? -> do not care, gurobi fixes
                return (self.model.x_flow[(a,f,p,t,s)]- self.model.x_flow[(a,f,p,t,ss)]==0) # TO DO: some slack here to improve feasibility? 
            else:
                return Constraint.Skip   # https://pyomo.readthedocs.io/en/stable/_modules/pyomo/core/base/constraint.html
        self.model.Nonanticipativity_x_Constr = Constraint(combinations(self.data.AFPT_S,self.data.S_SCENARIOS),rule = Nonanticipativity_x)

        def Nonanticipativity_b(model,i,j,m,r,f,v,t,s,ss):
            a = i,j,m,r
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.b_flow[(a,f,v,t,s)]- self.model.b_flow[(a,f,v,t,ss)]==0)
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_b_Constr = Constraint(combinations(self.data.AFVT_S,self.data.S_SCENARIOS),rule = Nonanticipativity_b)

        def Nonanticipativity_h(model,k,p,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.h_path[(k,p,t,s)]- self.model.h_path[(k,p,t,ss)]==0)
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_h_Constr = Constraint(combinations(self.data.KPT_S,self.data.S_SCENARIOS),rule = Nonanticipativity_h)

        def Nonanticipativity_h_bal(model,k,v,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.h_path_balancing[(k,v,t,s)]- self.model.h_path_balancing[(k,v,t,ss)]==0)
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_h_bal_Constr = Constraint(combinations(self.data.KVT_S,self.data.S_SCENARIOS),rule = Nonanticipativity_h_bal)

        def Nonanticipativity_stage(model,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.StageCosts[(t,s)]- self.model.StageCosts[(t,ss)]==0)
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_stage_Constr = Constraint(combinations(self.data.T_TIME_PERIODS_S,self.data.S_SCENARIOS),rule = Nonanticipativity_stage)

        def Nonanticipativity_eps(model,i,j,m,r,t,s,ss):
            e = (i,j,m,r)
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.epsilon_edge[(e,t,s)]- self.model.epsilon_edge[(e,t,ss)]==0)
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_eps_Constr = Constraint(combinations(self.data.ET_RAIL_S,self.data.S_SCENARIOS),rule = Nonanticipativity_eps)

        def Nonanticipativity_upg(model,i,j,m,r,f,t,s,ss):
            e = (i,j,m,r)
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.upsilon_upg[(e,f,t,s)]- self.model.upsilon_upg[(e,f,t,ss)]==0)
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_upg_Constr = Constraint(combinations(self.data.UT_UPG_S,self.data.S_SCENARIOS),rule = Nonanticipativity_upg)

        def Nonanticipativity_nu(model,n,c,m,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.nu_node[(n,c,m,t,s)]- self.model.nu_node[(n,c,m,t,ss)]==0)
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_nu_Constr = Constraint(combinations(self.data.NCMT_S,self.data.S_SCENARIOS),rule = Nonanticipativity_nu)

        def Nonanticipativity_y(model,i,j,m,r,f,t,s,ss):
            e = (i,j,m,r)
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.y_charge[(e,f,t,s)]- self.model.y_charge[(e,f,t,ss)]==0)
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_y_Constr = Constraint(combinations(self.data.EFT_CHARGE_S,self.data.S_SCENARIOS),rule = Nonanticipativity_y)

        def Nonanticipativity_z(model,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.z_emission[(t,s)]- self.model.z_emission[(t,ss)]==0)
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_z_Constr = Constraint(combinations(self.data.TS_S,self.data.S_SCENARIOS),rule = Nonanticipativity_z)

        def Nonanticipativity_totem(model,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.total_emissions[(t,s)]- self.model.total_emissions[(t,ss)]==0)
            else:
                return Constraint.Skip 
        self.model.Nonanticipativity_totem_Constr = Constraint(combinations(self.data.TS_S,self.data.S_SCENARIOS),rule = Nonanticipativity_totem)

        def Nonanticipativity_q(model,m,f,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.q_transp_amount[(m,f,t,s)]- self.model.q_transp_amount[(m,f,t,ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_q_Constr = Constraint(combinations(self.data.MFT_S,self.data.S_SCENARIOS),rule = Nonanticipativity_q)

        #this is different across scenarios!!
        #def Nonanticipativity_q_max(model,m,f,s,ss): 
        #    return (self.model.q_max_transp_amount[(m,f,s)]- self.model.q_max_transp_amount[(m,f,ss)]==0) 
        #self.model.Nonanticipativity_qmax_Constr = Constraint(combinations(self.data.MF_S,self.data.S_SCENARIOS),rule = Nonanticipativity_q_max)

        def Nonanticipativity_q_aux(model,m,f,t,s,ss):
            if (t in self.data.T_YEARLY_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.q_aux_transp_amount[(m,f,t,s)]- self.model.q_aux_transp_amount[(m,f,t,ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_qaux_Constr = Constraint(combinations(self.data.MFT_NEW_YEARLY_S,self.data.S_SCENARIOS),rule = Nonanticipativity_q_aux)
        
        def Nonanticipativity_qmode(model,m,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.q_mode_total_transp_amount[(m,t,s)]- self.model.q_mode_total_transp_amount[(m,t,ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_qmode_Constr = Constraint(combinations(self.data.MT_S,self.data.S_SCENARIOS),rule = Nonanticipativity_qmode)

        def Nonanticipativity_opex(model,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.TranspOpexCost[(t,s)]- self.model.TranspOpexCost[(t,ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_opex_Constr = Constraint(combinations(self.data.T_TIME_PERIODS_S,self.data.S_SCENARIOS),rule = Nonanticipativity_opex)

        def Nonanticipativity_co2(model,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.TranspCO2Cost[(t,s)]- self.model.TranspCO2Cost[(t,ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_co2_Constr = Constraint(combinations(self.data.T_TIME_PERIODS_S,self.data.S_SCENARIOS),rule = Nonanticipativity_co2)

        def Nonanticipativity_opexb(model,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.TranspOpexCostB[(t,s)]- self.model.TranspOpexCostB[(t,ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_opexb_Constr = Constraint(combinations(self.data.T_TIME_PERIODS_S,self.data.S_SCENARIOS),rule = Nonanticipativity_opexb)

        def Nonanticipativity_co2b(model,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.TranspCO2CostB[(t,s)]- self.model.TranspCO2CostB[(t,ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_co2b_Constr = Constraint(combinations(self.data.T_TIME_PERIODS_S,self.data.S_SCENARIOS),rule = Nonanticipativity_co2b)

        def Nonanticipativity_transf(model,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.TransfCost[(t,s)]- self.model.TransfCost[(t,ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_transf_Constr = Constraint(combinations(self.data.T_TIME_PERIODS_S,self.data.S_SCENARIOS),rule = Nonanticipativity_transf)

        def Nonanticipativity_edge(model,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.EdgeCost[(t,s)]- self.model.EdgeCost[(t,ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_edge_Constr = Constraint(combinations(self.data.T_TIME_PERIODS_S,self.data.S_SCENARIOS),rule = Nonanticipativity_edge)

        def Nonanticipativity_node(model,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.NodeCost[(t,s)]- self.model.NodeCost[(t,ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_node_Constr = Constraint(combinations(self.data.T_TIME_PERIODS_S,self.data.S_SCENARIOS),rule = Nonanticipativity_node)

        def Nonanticipativity_upgcost(model,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.UpgCost[(t,s)]- self.model.UpgCost[(t,ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_upgcost_Constr = Constraint(combinations(self.data.T_TIME_PERIODS_S,self.data.S_SCENARIOS),rule = Nonanticipativity_upgcost)

        def Nonanticipativity_chargecost(model,t,s,ss):
            if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                return (self.model.ChargeCost[(t,s)]- self.model.ChargeCost[(t,ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_chargecost_Constr = Constraint(combinations(self.data.T_TIME_PERIODS_S,self.data.S_SCENARIOS),rule = Nonanticipativity_chargecost)

        def Nonanticipativity_firststagecost(model,s,ss):
            if  (s is not ss): 
                return (self.model.FirstStageCosts[(s)]- self.model.FirstStageCosts[(ss)]==0) 
            else:
                return Constraint.Skip
        self.model.Nonanticipativity_firststagecost_Constr = Constraint([(s,ss) for s in self.data.S_SCENARIOS for ss in self.data.S_SCENARIOS],rule = Nonanticipativity_firststagecost)

        # self.model.MaxTranspPenaltyCost -> no non-anticipativity   
        # self.model.CvarAux   -> non-anticipativity already imposed in the model (not defined for scenarios)    
        # self.model.CvarPosPart -> Different across scenarios
        #self.model.SecondStageCosts  -> varies across scenarios


        #-----------------------------------------------#
        if NO_INVESTMENTS:
            for (i,j,m,r,tau,s) in self.data.ET_RAIL_S:
                self.model.epsilon_edge[i,j,m,r, tau,s].fix(0)
            for (e,f,t,s) in self.data.UT_UPG_S:
                self.model.upsilon_upg[(e,f,t,s )].fix(0)
            for (n,c,m,t,s ) in self.data.NCMT_S:
                self.model.nu_node[((n,c,m,t,s ))].fix(0)

        return self.model
    

    def fix_variables_first_stage(self,model_ev):  #this is the EV model that is input.
        
        # for (i,j,m,r,f,p,t) in self.data.AFPT_S:    
        #     a = (i,j,m,r)
        #     if t in self.data.T_TIME_FIRST_STAGE:
        #         self.model.x_flow[(a,f,p,t)].fix(0)
        
        base_scenario = 'BBB'

        #We quickly get infeasibilities 
        for (i,j,m,r,f,p,t,s) in self.data.AFPT_S:
            a = (i,j,m,r)
            self.model.x_flow[(a,f,p,t,s)].fixed = False
            if t in self.data.T_TIME_FIRST_STAGE:
                weight = model_ev.x_flow[(a,f,p,t,base_scenario)].value
                if weight is not None:
                    if weight > 0.05:
                        dev = 0.1
                        self.model.x_flow[(a,f,p,t,s)].setub((1+dev)*weight)
                        self.model.x_flow[(a,f,p,t,s)].setlb((1-dev)*weight)
                        #self.model.x_flow[(a,f,p,t,s)].fix(weight) 
                    #else:
                    #   self.model.x_flow[(a,f,p,t,s)].setlb(-1)
                    #   self.model.x_flow[(a,f,p,t,s)].setub(1)
                else:
                    pass #this does not happen

        for (i,j,m,r,t,s) in self.data.ET_RAIL_S:
            if t in self.data.T_TIME_FIRST_STAGE:
                e = (i,j,m,r)
                weight = model_ev.epsilon_edge[(e,t,base_scenario)].value
                if weight is not None:
                    if weight > 0.05:
                        self.model.epsilon_edge[(e,t,s)].fix(weight) 
                    else:
                        self.model.epsilon_edge[(e,t,s)].fix(0) 
        for (e,f,t,s) in self.data.UT_UPG_S:
            if t in self.data.T_TIME_FIRST_STAGE:
                weight = model_ev.upsilon_upg[(e,f,t,base_scenario)].value
                if weight is not None:
                    if weight > 0.05:
                        self.model.upsilon_upg[(e,f,t,s)].fix(weight) 
                    else:
                        self.model.upsilon_upg[(e,f,t,s)].fix(0)

        for (i,c,m,t,s) in self.data.NCMT_S:
            if t in self.data.T_TIME_FIRST_STAGE:
                weight = model_ev.nu_node[(i,c,m,t,base_scenario)].value
                if weight is not None:
                    if weight > 0.05:
                        self.model.nu_node[(i,c,m,t,s)].fix(weight)
                    else:
                        self.model.nu_node[(i,c,m,t,s)].fix(0)

        for (e,f,t,s) in self.data.EFT_CHARGE_S:
            if t in self.data.T_TIME_FIRST_STAGE:
                weight = model_ev.y_charge[(e,f,t,base_scenario)].value
                if weight is not None:
                    if weight > 0.05:
                        self.model.y_charge[(e,f,t,s)].fix(weight)
                    else:
                        self.model.y_charge[(e,f,t,s)].fix(0)
        
        
    def fix_variables_first_time_period(self,x_flow_base_period_init):
        
        # for v in solved_init_model.model.component_objects(pyo.Var, active=True):
        #     #print("Variable",v)  
        #     for index in v:
        #         if v[index].value is not None:
        #             #print ("   ",index, pyo.value(v[index]))
        #             var_big_model = getattr(self.model,str(v))
        #             var_big_model[index].fix(v[index].value)  

        #not providing a value in the fix operator leads to the following error:
        #TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'

        #fixing those leads to infeasibility? 
        for (i,j,m,r,f,p,t,s) in self.data.AFPT_S:
            a = (i,j,m,r)
            if (a,f,p,t,s) not in [(aa,ff,pp,tt,ss) for (aa,ff,pp,tt,ss,ww) in x_flow_base_period_init]:
                if t == self.data.T_TIME_PERIODS[0]:

                    self.model.x_flow[(a,f,p,t,s)].fix(0)   #seems to be a little bit faster

                    #self.model.x_flow[(a,f,p,t)].setlb(-1)
                    #self.model.x_flow[(a,f,p,t)].setub(5)
                    

        for (a,f,p,t,s,weight)in x_flow_base_period_init:
            
            #fixing this goes slower
            #self.model.x_flow[(a,f,p,t)].fix(weight)
            
            self.model.x_flow[(a,f,p,t,s)].fixed = False
            dev = 0.0001
            self.model.x_flow[(a,f,p,t,s)].setub((1+dev)*weight)
            self.model.x_flow[(a,f,p,t,s)].setlb((1-dev)*weight)
            

    def solve_model(self):  #GENERAL way to solve a single deterministic model

        self.results = self.opt.solve(self.model, tee=True, symbolic_solver_labels=True,
                                      keepfiles=True)  # , tee=True, symbolic_solver_labels=True, keepfiles=True)

        if (self.results.solver.status == pyomo.opt.SolverStatus.ok) and (
                self.results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal):
            print('the solution is feasible and optimal')
        elif self.results.solver.termination_condition == pyomo.opt.TerminationCondition.infeasible:
            print('the model is infeasible')
            #log_infeasible_constraints(self.model,log_expression=True, log_variables=True)
            #logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)
            #print(value(model.z))

        else:
            print('Solver Status: '), self.results.solver.status
            print('Termination Condition: '), self.results.solver.termination_condition

        print('Solution time: ' + str(self.results.solver.time))
        


        
