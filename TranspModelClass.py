#Pyomo
import pyomo.opt   # we need SolverFactory,SolverStatus,TerminationCondition
import pyomo.opt.base as mobase
from pyomo.environ import *
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
import logging
from Data.settings import *
from Data.Create_Sets_Class import TransportSets   #FreightTransportModel.Data.Create_Sets_Class
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
        self.data = data
        self.risk_info = risk_info # stores parameters of the risk measure (i.e., lambda and alpha for mean-CVaR)

        self.calculate_max_transp_amount_exact = False

    def construct_model(self):
        
        #Significant speed improvements can be obtained using the LinearExpression object when there are long, dense, linear expressions.
        # USe linearexpressions: https://pyomo.readthedocs.io/en/stable/advanced_topics/linearexpression.html
        

        "VARIABLES"
        # Binary, NonNegativeReals, PositiveReals, etc

        self.model.x_flow = Var(self.data.AFPT, within=NonNegativeReals)
        self.model.b_flow = Var(self.data.AFVT, within=NonNegativeReals)
        self.model.h_path = Var(self.data.KPT, within=NonNegativeReals)# flow on paths K,p
        self.model.h_path_balancing = Var(self.data.KVT, within=NonNegativeReals)# flow on paths K,p

        self.model.StageCosts = Var(self.data.T_TIME_PERIODS, within = NonNegativeReals)
        
        self.model.epsilon_edge = Var(self.data.ET_RAIL, within = Binary) #within = Binary
        self.model.upsilon_upg = Var(self.data.UT_UPG, within = Binary) #bin.variable for investments upgrade/new infrastructure u at link l, time period t
        self.model.nu_node = Var(self.data.NCMT, within = Binary) #step-wise investment in terminals
        self.model.y_charge = Var(self.data.EFT_CHARGE, within=NonNegativeReals)

        self.model.z_emission = Var(self.data.TS, within = NonNegativeReals)

        self.model.total_emissions = Var(self.data.TS, within=NonNegativeReals) #instead of T_PERIODS!
                                        # bounds=emission_bound)  # just a variable to help with output
        
        self.model.q_transp_amount = Var(self.data.MFT, within=NonNegativeReals)
        self.model.q_max_transp_amount = Var(self.data.MF, within=NonNegativeReals)

        MAX_TRANSP_PENALTY = MAX_TRANSPORT_AMOUNT_PENALTY
        if self.calculate_max_transp_amount_exact:
            self.model.chi_max_transp = Var(self.data.MFT, within = Binary)
            MAX_TRANSP_PENALTY = 0

        #COST_VARIABLES

        self.model.TranspOpexCost = Var(self.data.T_TIME_PERIODS, within=NonNegativeReals)
        def TranspOpexCost(model, t):
            return (self.model.TranspOpexCost[t] == sum((self.data.C_TRANSP_COST[(i,j,m,r,f,p,t)])*self.model.x_flow[(i,j,m,r,f,p,t)] 
                                      for p in self.data.P_PRODUCTS for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m]) )
        self.model.TranspOpexCostConstr = Constraint(self.data.T_TIME_PERIODS, rule=TranspOpexCost)
        
        self.model.TranspCO2Cost = Var(self.data.T_TIME_PERIODS, within=NonNegativeReals)
        def TranspCO2Cost(model, t):
            return (self.model.TranspCO2Cost[t] == sum((self.data.C_CO2[(i,j,m,r,f,p,t)])*self.model.x_flow[(i,j,m,r,f,p,t)] 
                                      for p in self.data.P_PRODUCTS for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m]) )
        self.model.TranspCO2CostConstr = Constraint(self.data.T_TIME_PERIODS, rule=TranspCO2Cost)
        
        self.model.TranspOpexCostB = Var(self.data.T_TIME_PERIODS, within=NonNegativeReals)
        def TranspOpexCostB(model, t):
            return (self.model.TranspOpexCostB[t] == sum( EMPTY_VEHICLE_FACTOR*(self.data.C_TRANSP_COST[(i,j,m,r,f,self.data.cheapest_product_per_vehicle[(m,f,t,v)],t)]) * self.model.b_flow[(i,j,m,r,f,v,t)] 
                                    for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m] )  )
        self.model.TranspOpexCostBConstr = Constraint(self.data.T_TIME_PERIODS, rule=TranspOpexCostB)
        
        self.model.TranspCO2CostB = Var(self.data.T_TIME_PERIODS, within=NonNegativeReals)
        def TranspCO2CostB(model, t):
            return (self.model.TranspCO2CostB[t] == sum( EMPTY_VEHICLE_FACTOR*(self.data.C_CO2[(i,j,m,r,f,self.data.cheapest_product_per_vehicle[(m,f,t,v)],t)]) * self.model.b_flow[(i,j,m,r,f,v,t)] 
                                    for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m] )  )
        self.model.TranspCO2CostBConstr = Constraint(self.data.T_TIME_PERIODS, rule=TranspCO2CostB)
        
        self.model.TransfCost = Var(self.data.T_TIME_PERIODS, within=NonNegativeReals)
        def TransfCost(model, t):
            return (self.model.TransfCost[t] == sum(self.data.C_TRANSFER[(k,p)]*self.model.h_path[k,p,t] for p in self.data.P_PRODUCTS  for k in self.data.MULTI_MODE_PATHS) )
        self.model.TransfCostConstr = Constraint(self.data.T_TIME_PERIODS, rule=TransfCost)
        
        self.model.EdgeCost = Var(self.data.T_TIME_PERIODS, within=NonNegativeReals)
        def EdgeCost(model, t):
            return (self.model.EdgeCost[t] == sum(self.data.C_EDGE_RAIL[e]*self.model.epsilon_edge[(e,t)] for e in self.data.E_EDGES_RAIL) )
        self.model.EdgeCostConstr = Constraint(self.data.T_TIME_PERIODS, rule=EdgeCost)
        
        self.model.NodeCost = Var(self.data.T_TIME_PERIODS, within=NonNegativeReals)
        def NodeCost(model, t):
            return (self.model.NodeCost[t] == sum(self.data.C_NODE[(i,c,m)]*self.model.nu_node[(i,c,m,t)] for (i, m) in self.data.NM_LIST_CAP for c in self.data.TERMINAL_TYPE[m]) )
        self.model.NodeCostConstr = Constraint(self.data.T_TIME_PERIODS, rule=NodeCost)
        
        self.model.UpgCost = Var(self.data.T_TIME_PERIODS, within=NonNegativeReals)
        def UpgCost(model, t):
            return (self.model.UpgCost[t] == sum(self.data.C_UPG[(e,f)]*self.model.upsilon_upg[(e,f,t)] for (e,f) in self.data.U_UPGRADE))
        self.model.UpgCostConstr = Constraint(self.data.T_TIME_PERIODS, rule=UpgCost)
        
        self.model.ChargeCost = Var(self.data.T_TIME_PERIODS, within=NonNegativeReals)
        def ChargeCost(model, t):
            return (self.model.ChargeCost[t] == sum(self.data.C_CHARGE[(e,f)]*self.model.y_charge[(e,f,t)] for (e,f) in self.data.EF_CHARGING) )
        self.model.ChargeCostConstr = Constraint(self.data.T_TIME_PERIODS, rule=ChargeCost)
        
        self.model.MaxTranspPenaltyCost = Var(within=NonNegativeReals)
        def MaxTranspPenaltyCost(model): #TEST
            return (self.model.MaxTranspPenaltyCost == MAX_TRANSP_PENALTY*sum(self.model.q_max_transp_amount[m,f] for m in self.data.M_MODES for f in self.data.FM_FUEL[m]) )
        self.model.MaxTranspPenaltyCostConstr = Constraint(rule=MaxTranspPenaltyCost)


        # CVaR variables
        
        # CVaR auxiliary variable (corresponds to u)
        self.model.CvarAux = Var(within=Reals)

        #CVaR positive part variable (corresponds to z = (f - u)^+)
        self.model.CvarPosPart = Var(within=NonNegativeReals)               #note: includes one positive part constraint: z \geq 0


        "OBJECTIVE"
        #-------------------------

        def StageCostsVar(model, t):  #TODO: QUESTION: WHY DO WE USE "model" as input instead of "self"? (not just here, but everywhere)

            # SOME QUICK TESTING SHOWED THAT SUM_PRODUCT IS QUITE A BIT SLOWER THAN SIMPLY TAKING THE SUM...
            yearly_transp_cost = (self.model.TranspOpexCost[t] + self.model.TranspCO2Cost[t] + self.model.TranspOpexCostB[t] + self.model.TranspCO2CostB[t]) 
            
            #sum(base_data.D_DISCOUNT_RATE**n for n in list(range(10)))
            if t == self.data.T_TIME_PERIODS[-1]:
                factor = self.data.D_DISCOUNT_RATE**self.data.Y_YEARS[t][0] * (1/(1-self.data.D_DISCOUNT_RATE))      #was 2.77, becomes 9
            else:
                factor = sum(self.data.D_DISCOUNT_RATE**n for n in self.data.Y_YEARS[t])
            opex_costs = factor*(yearly_transp_cost+self.model.TransfCost[t]) 
            
            delta = self.data.D_DISCOUNT_RATE**self.data.Y_YEARS[t][0]
            investment_costs = self.model.EdgeCost[t] + self.model.NodeCost[t] + self.model.UpgCost[t] + self.model.ChargeCost[t] 
            
            return (self.model.StageCosts[t] == (opex_costs + delta*investment_costs + EMISSION_VIOLATION_PENALTY*self.model.z_emission[t]))
        self.model.stage_costs = Constraint(self.data.T_TIME_PERIODS, rule = StageCostsVar)
        
        # scenario objective value variable (risk-neutral model would take expectation over this in the objective)
        self.model.ScenObjValue = Var(within=Reals)
        def ScenObjValue(model):
            return self.model.ScenObjValue == sum(self.model.StageCosts[t] for t in self.data.T_TIME_PERIODS) +  self.model.MaxTranspPenaltyCost  # corresponds to f(x,\xi)
        self.model.ScenObjValueConstr = Constraint(rule = ScenObjValue)
        
        #CVaR positive part constraint
        def CvarRule(model):
            return self.model.CvarPosPart >= self.model.ScenObjValue - self.model.CvarAux       # z \geq f - u
        self.model.CvarPosPartConstr = Constraint(rule=CvarRule) # add to model

        # mean-CVaR:
        def objfun(model):          
            # scenario objective value for risk-averse (mean-CVaR) model:
            mean_cvar_scen_obj_value = ( (1 - self.risk_info.cvar_coeff) * self.model.ScenObjValue                                                            # expectation part
                                       + self.risk_info.cvar_coeff * (self.model.CvarAux + (1 - self.risk_info.cvar_alpha)**(-1) * self.model.CvarPosPart) )  # CVaR part
            return mean_cvar_scen_obj_value

        # risk-neutral #NOTE: TEMPORARY
        def objfun_risk_neutral(model):          
            # scenario objective value for risk-neutral model:
            risk_neutral_scen_obj_value = self.model.ScenObjValue
            return risk_neutral_scen_obj_value


        # give objective function to model
        #self.model.Obj = Objective(rule=objfun, sense=minimize)
        self.model.Obj = Objective(rule=objfun_risk_neutral, sense=minimize) #TEMPORARY: risk-neutral
        

        ###########
        #"""
        # NEW: risk-averse
        # risk measure: \rho(.) = (1 - \lambda) * E[.] + \lambda * CVaR_\alpha(.)
        # so \rho(f(x,\xi)) = \min_u \E[ (1 - \lambda) * f + \lambda * (u + (1 - \alpha)**(-1) * z) ]
        # with
        # z \geq f - u
        # z \geq 0
        
        # CVaR parameters:
        #cvar_coeff = 0.20 # corresponds to \lambda (coefficient for CVaR, relative importance of CVaR). Set at e.g. 20%
        #cvar_alpha = 0.10 # corresponds to \alpha (how far in the tail we're looking). Set at e.g. 10% (i.e., three worst scenarios)
        
        # CVaR variables:
        #cvar_aux # corresponds to u (auxiliary variable for CVaR)
        #cvar_pp # corresponds to z (positive part in CVaR)
        
        # CVaR constraints:
        #cvar_pp >= f - cvar_aux     # z \geq f - u
        #cvar_pp >= 0                # z \geq 0

        # mean-CVaR objective: 
        #(1 - cvar_coeff) * f + cvar_coeff * (cvar_aux + (1 - cvar_alpha)**(-1) * cvar_pp) # replace "f" by this

        #"""
        
        ###########


        "CONSTRAINTS"

        # DEMAND
                
        def FlowRule(model, o, d, p, t):
            return sum(self.model.h_path[(k, p, t)] for k in self.data.OD_PATHS[(o, d)]) >= self.data.D_DEMAND[
                (o, d, p, t)]
        # THIS SHOULD BE AN EQUALITY; BUT THEN THE PROBLEM GETS EASIER WITH A LARGER THAN OR EQUAL
        self.model.Flow = Constraint(self.data.ODPTS, rule=FlowRule)
        

        # PATHFLOW

        def PathArcRule(model, i, j, m, r, p, t):
            a= (i,j,m,r)
            return sum(self.model.x_flow[a, f, p, t] for f in self.data.FM_FUEL[m]) == sum(
                self.model.h_path[k, p, t] for k in self.data.KA_PATHS[a] )
        self.model.PathArcRel = Constraint(self.data.APT, rule=PathArcRule)

        def PathArcRuleBalancing(model, i, j, m, r, v, t):
            a= (i,j,m,r)
            return sum(self.model.b_flow[a, f, v, t] for f in self.data.FM_FUEL[m]) == sum(
                self.model.h_path_balancing[k, v, t] for k in self.data.KA_PATHS_UNIMODAL[a] )
        self.model.PathArcRelBalance = Constraint(self.data.AVT, rule=PathArcRuleBalancing)

        # FLEET BALANCING

        def FleetBalance(model, n,m,f,v, t):
            disbalance_in_node = (sum(self.model.x_flow[(a, f, p, t)] for a in self.data.ANM_ARCS_IN[(n,m)] for p in self.data.PV_PRODUCTS[v]) - 
                    sum(self.model.x_flow[(a, f, p, t)] for a in self.data.ANM_ARCS_OUT[(n,m)] for p in self.data.PV_PRODUCTS[v]))  
            empty_trips = (sum(self.model.b_flow[(a, f, v, t)] for a in self.data.ANM_ARCS_OUT[(n,m)]) -
                        sum(self.model.b_flow[(a, f, v, t)] for a in self.data.ANM_ARCS_IN[(n,m)]))            
            return (disbalance_in_node == empty_trips)
        self.model.FleetBalance = Constraint(self.data.NMFVT, rule=FleetBalance)

        #-----------------------------------------------#

        # EMISSIONS
        def emissions_rule(model, t):
            return (
                self.model.total_emissions[t] == sum(
                self.data.E_EMISSIONS[i,j,m,r,f, p, t] * self.model.x_flow[i,j,m,r,f, p, t] for p in self.data.P_PRODUCTS
                for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m]) + 
                sum(self.data.E_EMISSIONS[i,j,m,r,f, self.data.cheapest_product_per_vehicle[(m,f,t,v)], t]*EMPTY_VEHICLE_FACTOR * self.model.b_flow[i,j,m,r,f, v, t] 
                    for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m])
                )
        
        self.model.Emissions = Constraint(self.data.TS, rule=emissions_rule) #removed self.data.T_TIME_PERIODS

        # Emission limit
        def EmissionCapRule(model, t):
            return self.model.total_emissions[t] <= self.data.CO2_CAP[t]/100*self.model.total_emissions[self.data.T_TIME_PERIODS[0]] + self.model.z_emission[t]
        self.model.EmissionCap = Constraint(self.data.TS, rule=EmissionCapRule)
        
        #-----------------------------------------------#
        
        #CAPACITY
        def CapacitatedFlowRule(model,i,j,m,r,ii,jj,mm,rr,t):
            e = (i,j,m,r)
            a = (ii,jj,mm,rr)
            return (sum(self.model.x_flow[a, f, p, t] for p in self.data.P_PRODUCTS for f in self.data.FM_FUEL[m]) + 
                    sum(self.model.b_flow[a, f, v, t] for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m] ) <= 0.5*(self.data.Q_EDGE_BASE_RAIL[e] +
                   + self.data.Q_EDGE_RAIL[e] * sum(self.model.epsilon_edge[e, tau] for tau in self.data.T_TIME_PERIODS if tau <= (t-self.data.LEAD_TIME_EDGE_RAIL[e]))))
        self.model.CapacitatedFlow = Constraint(self.data.EAT_RAIL, rule = CapacitatedFlowRule)
        
        #Num expansions
        def ExpansionLimitRule(model,i,j,m,r):
            e = (i,j,m,r)
            return (sum(self.model.epsilon_edge[(e,t)] for t in self.data.T_TIME_PERIODS_NOT_NOW) <= 1)
        if len(self.data.T_TIME_PERIODS)>1:
            self.model.ExpansionCap = Constraint(self.data.E_EDGES_RAIL, rule = ExpansionLimitRule)
        
        #Terminal capacity constraint. We keep the old notation here, so we can distinguish between OD and transfer, if they take up different capacity.
        def TerminalCapRule(model, i, c, m,t):
            return(sum(self.model.h_path[k, p, t] for k in self.data.ORIGIN_PATHS[(i,m)] for p in self.data.PT[c]) + 
                   sum(self.model.h_path[k, p, t] for k in self.data.DESTINATION_PATHS[(i,m)] for p in self.data.PT[c]) +
                   sum(self.model.h_path[k,p,t] for k in self.data.TRANSFER_PATHS[(i,m)] for p in self.data.PT[c]) <= 
                   self.data.Q_NODE_BASE[i,c,m]+self.data.Q_NODE[i,c,m]*sum(self.model.nu_node[i,c,m,tau] for tau in self.data.T_TIME_PERIODS_NOT_NOW if tau <= (t-self.data.LEAD_TIME_NODE[i,c,m])))
        self.model.TerminalCap = Constraint(self.data.NCMT, rule = TerminalCapRule)
        
        #Num expansions of terminal NEW -- how many times you can perform a step-wise increase of the capacity
        def TerminalCapExpRule(model, i, c,m):
            return(sum(self.model.nu_node[i,c,m,t] for t in self.data.T_TIME_PERIODS_NOT_NOW) <= 1) # THIS WAS AT 4 .... self.data.INV_NODE[i,m,c])
        if len(self.data.T_TIME_PERIODS)>1:
            self.model.TerminalCapExp = Constraint(self.data.NCM, rule = TerminalCapExpRule)

        #Charging / Filling
        def ChargingCapArcRule(model, i, j, m, r,f, t):
            e = (i, j, m, r)
            return (sum(self.model.x_flow[a,f,p, t] for p in self.data.P_PRODUCTS
                       for a in self.data.AE_ARCS[e]) + sum(self.model.b_flow[a,f,v, t] for a in self.data.AE_ARCS[e] 
                        for v in self.data.VEHICLE_TYPES_M[m]) <= self.data.Q_CHARGE_BASE[(e,f)] +
                   sum(self.model.y_charge[(e,f,tau)] for tau in self.data.T_TIME_PERIODS_NOT_NOW if tau <= (t-self.data.LEAD_TIME_CHARGING[(e,f)])))
        self.model.ChargingCapArc = Constraint(self.data.EFT_CHARGE, rule=ChargingCapArcRule)
        #AIM also looked into charging infrastructure in NODES

        #Upgrading
        def InvestmentInfraRule(model,i,j,m,r,f,t):
            e = (i,j,m,r)
            return (sum(self.model.x_flow[a,f,p,t] for p in self.data.P_PRODUCTS for a in self.data.AE_ARCS[e])
                    <= self.data.BIG_M_UPG[e]*sum(self.model.upsilon_upg[e,f,tau] for tau in self.data.T_TIME_PERIODS_NOT_NOW if tau <= (t-self.data.LEAD_TIME_UPGRADE[(e,f)])))
        self.model.InvestmentInfra = Constraint(self.data.UT_UPG, rule = InvestmentInfraRule)
    
        #-----------------------------------------------#
    
        #TransportArbeid
        def TotalTransportAmountRule(model,m,f,t):
            return (self.model.q_transp_amount[m,f,t] == sum( self.data.AVG_DISTANCE[a]*self.model.x_flow[a,f,p,t] for p in self.data.P_PRODUCTS 
                                                            for a in self.data.AM_ARCS[m]))
        self.model.TotalTranspAmount = Constraint(self.data.MFT, rule = TotalTransportAmountRule)
        
        #Technology maturity limit
        def TechMaturityLimitRule(model, m, f, t):
            return (self.model.q_transp_amount[(m,f,t)] <= self.data.R_TECH_READINESS_MATURITY[(m,f,t)]/100*sum(self.model.q_transp_amount[(m,ff,t)] for ff in self.data.FM_FUEL[m]))   #TO DO: CHANGE THIS Q_TECH to R*M
        self.model.TechMaturityLimit = Constraint(self.data.MFT_MATURITY, rule = TechMaturityLimitRule)

        #Initialize the transport amounts (put an upper bound at first)
        def InitTranspAmountRule(model, m, f, t):
            return (self.model.q_transp_amount[(m,f,t)] <= self.data.Q_SHARE_INIT_MAX[(m,f,t)]/100*sum(self.model.q_transp_amount[(m,ff,t)] for ff in self.data.FM_FUEL[m]))   #TO DO: CHANGE THIS Q_TECH to R*M
        self.model.InitTranspAmount = Constraint(self.data.MFT_INIT_TRANSP_SHARE, rule = InitTranspAmountRule)
        

        #Max TransportArbeid
        def MaxTransportAmountRule(model,m,f,t):
            return (self.model.q_max_transp_amount[m,f] >= self.model.q_transp_amount[m,f,t])
        self.model.MaxTranspAmount = Constraint(self.data.MFT, rule = MaxTransportAmountRule)
        
        if self.calculate_max_transp_amount_exact:
            def MaxTransportAmountRule2(model,m,f,t):
                M = np.max(list(self.data.AVG_DISTANCE.values())) * self.data.D_DEMAND_AGGR[t]  #maybe we can use mean as well
                return (self.model.q_max_transp_amount[m,f] <= self.model.q_transp_amount[m,f,t] + M*(1-self.model.chi_max_transp[m,f,t]))
            self.model.MaxTranspAmount2 = Constraint(self.data.MFT, rule = MaxTransportAmountRule2)

            def MaxTransportAmountRule3(model,m,f):
                return (sum(self.model.chi_max_transp[m,f,t] for t in self.data.T_TIME_PERIODS) == 1)
            self.model.MaxTranspAmount3 = Constraint(self.data.MF, rule = MaxTransportAmountRule3)
            
        #Fleet Renewal
        def FleetRenewalRule(model,m,f, t):
            decrease = self.model.q_transp_amount[(m,f,self.data.T_MIN1[t])] - self.model.q_transp_amount[(m,f,t)]
            factor = (t - self.data.T_MIN1[t]) / self.data.LIFETIME[(m,f)]
            return (decrease <= factor*self.model.q_max_transp_amount[m,f])
        self.model.FleetRenewal = Constraint(self.data.MFT_MIN0, rule = FleetRenewalRule)
        
        #-----------------------------------------------#
        #    Specific constraints
        #-----------------------------------------------#

        #Do not use too much road
        def RoadDevelopmentRule(model, t):
            m = "Road"
            return (sum(self.model.q_transp_amount[(m,f,t)] for f in self.data.FM_FUEL[m]) <= \
                GROWTH_ON_ROAD*sum(self.model.q_transp_amount[(m,f,self.data.T_TIME_PERIODS[0])] for f in self.data.FM_FUEL[m]))   
        self.model.RoadDevelopment = Constraint(self.data.T_TIME_PERIODS, rule = RoadDevelopmentRule)

        def InitialModeSplitLower(model,m):
            t0 = self.data.T_TIME_PERIODS[0]
            total_transport_amount = sum(self.model.q_transp_amount[(mm,f,t0)] for mm in self.data.M_MODES for f in self.data.FM_FUEL[mm])
            modal_transport_amount = sum(self.model.q_transp_amount[(m,f,t0)] for f in self.data.FM_FUEL[m])
            return ( INIT_MODE_SPLIT_LOWER[m]*total_transport_amount <= modal_transport_amount ) 
        self.model.InitModeSplitLower = Constraint(self.data.M_MODES,rule = InitialModeSplitLower)

        def InitialModeSplitUpper(model,m):
            t0 = self.data.T_TIME_PERIODS[0]
            total_transport_amount = sum(self.model.q_transp_amount[(mm,f,t0)] for mm in self.data.M_MODES for f in self.data.FM_FUEL[mm])
            modal_transport_amount = sum(self.model.q_transp_amount[(m,f,t0)] for f in self.data.FM_FUEL[m])
            return (modal_transport_amount <= INIT_MODE_SPLIT_UPPER[m]*total_transport_amount) 
        self.model.InitModeSplitUpper = Constraint(self.data.M_MODES,rule = InitialModeSplitUpper)

        #-----------------------------------------------#



        if NO_INVESTMENTS:
            for (i,j,m,r,tau) in self.data.ET_RAIL:
                self.model.epsilon_edge[i,j,m,r, tau].fix(0)
            for (e,f,t) in self.data.UT_UPG:
                self.model.upsilon_upg[(e,f,t)].fix(0)
            for (n,c,m,t) in self.data.NCMT:
                self.model.nu_node[((n,c,m,t))].fix(0)

        return self.model
    

    def fix_variables_first_stage(self,output):
        
        for index,row in output.all_variables[output.all_variables['time_period'].isin(self.data.T_TIME_FIRST_STAGE)].iterrows():
            variable = row['variable']
            i = row['from']
            j = row['to']
            m = row['mode']
            r = row['route']
            a = (i,j,m,r)
            e = (i,j,m,r)
            f = row['fuel']
            p = row['product']
            t = row['time_period']
            w = row['weight']
            #s = row['scenario'] #not used
            v = row['vehicle_type']
            k = row['path']
            c = row['terminal_type']
            if variable == 'x_flow':
                self.model.x_flow[(a,f,p,t)].fix(w)
            elif variable == 'b_flow':
                self.model.b_flow[(a,f,v,t)].fix(w)
            elif variable == 'h_path':
                self.model.h_path[(k,p,t)].fix(w)
            elif variable == 'epsilon_edge':
                self.model.epsilon_edge[(e,t)].fix(w)
            elif variable == 'upsilon_upg':
                self.model.upsilon_upg[(i,j,m,r,f,t)].fix(w)
            elif variable == 'nu_node':
                self.model.nu_node[(i, c, m, t)].fix(w)
            elif variable == 'y_charging':
                self.model.y_charge[(i,j,m,r,f,t)].fix(w)
            elif variable == 'z_emission':
                self.model.z_emission[t].fix(w)
            elif variable == 'total_emissions':
                self.model.total_emissions[t].fix(w)
            elif variable == 'q_transp_amount':
                self.model.q_transp_amount[(m, f, t)].fix(w)
            elif variable == 'q_max_transp_amount':
                self.model.q_max_transp_amount[(m, f)].fix(w)

    def fix_variables_first_time_period(self,solved_init_model):
        
        # for v in solved_init_model.model.component_objects(pyo.Var, active=True):
        #     #print("Variable",v)  
        #     for index in v:
        #         if v[index].value is not None:
        #             #print ("   ",index, pyo.value(v[index]))
        #             var_big_model = getattr(self.model,str(v))
        #             var_big_model[index].fix(v[index].value)  

        #not providing a value in the fix operator leads to the following error:
        #TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'

        for j in solved_init_model.model.q_transp_amount:
            val = solved_init_model.model.q_transp_amount[j].value
            if val >= 0:
                self.model.q_transp_amount[j].fix(val)

        for j in solved_init_model.model.b_flow:
            val = solved_init_model.model.b_flow[j].value
            if val >= 0:
                self.model.b_flow[j].fix(val)

        for j in solved_init_model.model.x_flow:
            val = solved_init_model.model.x_flow[j].value
            if val >= 0:
                self.model.x_flow[j].fix(val)
        
        for j in solved_init_model.model.h_path:
            val = solved_init_model.model.h_path[j].value
            if val >= 0:
                self.model.h_path[j].fix(val)

        for j in solved_init_model.model.total_emissions:
            val = solved_init_model.model.total_emissions[j].value
            if val >= 0:
                self.model.total_emissions[j].fix(val)


    def solve_model(self):

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
        


        
