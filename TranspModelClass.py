#Pyomo
import pyomo.opt   # we need SolverFactory,SolverStatus,TerminationCondition
import pyomo.opt.base as mobase
from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints
import logging
from Data.settings import *
from Data.Create_Sets_Class import TransportSets   #FreightTransportModel.Data.Create_Sets_Class
import os
os.getcwd()

############# Class ################



class TranspModel:

    def __init__(self, data):

        self.instance = instance
        #timelimit in minutes, etc
        #elf.maturity_scenario = maturity_scenario

        self.results = 0  # results is an structure filled out later in solve_model()
        self.status = ""  # status is a string filled out later in solve_model()
        self.model = ConcreteModel()
        self.opt = pyomo.opt.SolverFactory('gurobi') #gurobi
        self.results = 0  # results is an structure filled out later in solve_model()
        self.status = ""  # status is a string filled out later in solve_model()
        self.scenario = scenario
        self.carbon_scenario = carbon_scenario
        self.fuel_costs = fuel_costs
        self.emission_reduction = emission_reduction
        #IMPORT THE DATA
        self.data = data
        self.data.update_parameters(scenario, carbon_scenario, fuel_costs, emission_reduction)


    def construct_model(self):
        
        #Significant speed improvements can be obtained using the LinearExpression object when there are long, dense, linear expressions.
        # USe linearexpressions: https://pyomo.readthedocs.io/en/stable/advanced_topics/linearexpression.html
        

        "VARIABLES"
        # Binary, NonNegativeReals, PositiveReals, etc

        self.model.x_flow = Var(self.data.AFPT, within=NonNegativeReals)
        self.model.b_flow = Var(self.data.AFVT, within=NonNegativeReals)
        self.model.h_flow = Var(self.data.KPT, within=NonNegativeReals)# flow on paths K,p
        self.model.h_flow_balancing = Var(self.data.KPT, within=NonNegativeReals)# flow on paths K,p

        self.model.StageCosts = Var(self.data.T_TIME_PERIODS, within = NonNegativeReals)
        
        self.model.v_edge = Var(self.data.ET_RAIL, within = Binary) #within = Binary
        self.model.u_upg = Var(self.data.UT_UPG, within = Binary) #bin.variable for investments upgrade/new infrastructure u at link l, time period t
        self.model.w_node = Var(self.data.NCMT, within = Binary) #step-wise investment in terminals
        
        self.model.y_charge = Var(self.data.EFT_CHARGE, within=NonNegativeReals)
        self.model.z_emission = Var(self.data.TS, within = NonNegativeReals)

        self.model.total_emissions = Var(self.data.TS, within=NonNegativeReals) #instead of T_PERIODS!
                                        # bounds=emission_bound)  # just a variable to help with output
        
        self.model.q_transp_amount = Var(self.data.MFT, within=NonNegativeReals)
        self.model.q_max_transp_amount = Var(self.data.MFT, within=NonNegativeReals)

        # self.model.AFPT = Set(initialize=self.data.AFPT)
        # list(self.model.AFPT)
        # self.model.C_TRANSP_COST = Param(self.model.AFPT, initialize=self.data.C_TRANSP_COST) #, default=0
        # self.model.C_CO2 = Param(self.model.AFPT, initialize=self.data.C_CO2) #, default=0

        "OBJECTIVE"
        #TO DO: check how the CO2 kicks in? Could be baked into C_TRANSP_COST
        def StageCostsVar(model, t):
            # SOME QUICK TESTING SHOWED THAT SUM_PRODUCT IS QUITE A BIT SLOWER THAN SIMPLY TAKING THE SUM...
            # yearly_transp_cost = (sum_product(self.model.C_TRANSP_COST ,self.model.x_flow,
            #                                  index=[(i,j,m,r,f,p,tt) for (i,j,m,r,f,p,tt) in self.data.AFPT if tt==t]) +
            #                       sum_product(self.model.C_CO2 ,self.model.x_flow,index=[(i,j,m,r,f,p,tt) for (i,j,m,r,f,p,tt) in self.data.AFPT if tt==t]))
            yearly_transp_cost = sum((self.data.C_TRANSP_COST[(i,j,m,r,f,p,t)]+self.data.C_CO2[(i,j,m,r,f,p,t)])*(self.model.x_flow[(i,j,m,r,f,p,t)]+self.model.b_flow[(i,j,m,r,f,p,t)]) 
                                      for p in self.data.P_PRODUCTS for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m])
            yearly_transfer_cost = sum(self.data.C_TRANSFER[(k,p)]*self.model.h_flow[k,p,t] for p in self.data.P_PRODUCTS  for k in self.data.MULTI_MODE_PATHS)
            delta = self.data.D_DISCOUNT_RATE**self.data.Y_YEARS[t][0]
            return(self.model.StageCosts[t] == (
                sum(self.data.D_DISCOUNT_RATE**n*(yearly_transp_cost+yearly_transfer_cost) for n in self.data.Y_YEARS[t]) + 
                sum(delta*self.data.C_EDGE_RAIL[e]*self.model.v_edge[(e,t)] for e in self.data.E_EDGES_RAIL) +
                sum(delta*self.data.C_NODE[(i,c,m)]*self.model.w_node[(i,c,m,t)] for (i, m) in self.data.NM_LIST_CAP for c in self.data.TERMINAL_TYPE[m] ) +
                # maybe N_NODES_CAP_NORWAY is needed?
                sum(delta*self.data.C_UPG[(e,f)]*self.model.u_upg[(e,f,t)] for (e,f) in self.data.U_UPGRADE) +
                sum(delta*self.data.C_CHARGE[(e,f)]*self.model.y_charge[(e,f,t)] for (e,f) in self.data.EF_CHARGING) +
                EMISSION_VIOLATION_PENALTY*self.model.z_emission[t] +
                MAX_TRANSPORT_AMOUNT_PENALTY*sum(self.model.q_max_transp_amount[m,f,t] for m in self.data.M_MODES for f in self.data.FM_FUEL[m]) 
                ))
        self.model.stage_costs = Constraint(self.data.T_TIME_PERIODS, rule = StageCostsVar)
        
        def objfun(model):
            obj_value = sum(self.model.StageCosts[t] for t in self.data.T_TIME_PERIODS) 
            return obj_value
        self.model.Obj = Objective(rule=objfun, sense=minimize)

        "CONSTRAINTS"

        # DEMAND
                
        def FlowRule(model, o, d, p, t):
            return sum(self.model.h_flow[(k, p, t)] for k in self.data.OD_PATHS[(o, d)]) >= self.data.D_DEMAND[
                (o, d, p, t)]
        # THIS SHOULD BE AN EQUALITY; BUT THEN THE PROBLEM GETS EASIER WITH A LARGER THAN OR EQUAL
        self.model.Flow = Constraint(self.data.ODPTS, rule=FlowRule)
        

        # PATHFLOW

        def PathArcRule(model, i, j, m, r, p, t):
            a= (i,j,m,r)
            return sum(self.model.x_flow[a, f, p, t] for f in self.data.FM_FUEL[m]) == sum(
                self.model.h_flow[k, p, t] for k in self.data.KA_PATHS[a] )
        self.model.PathArcRel = Constraint(self.data.APT, rule=PathArcRule)

        def PathArcRuleBalancing(model, i, j, m, r, p, t):
            a= (i,j,m,r)
            return sum(self.model.b_flow[a, f, p, t] for f in self.data.FM_FUEL[m]) == sum(
                self.model.h_flow_balancing[k, p, t] for k in self.data.KA_PATHS_UNIMODAL[a] )
        self.model.PathArcRelBalance = Constraint(self.data.APT, rule=PathArcRuleBalancing)

        # FLEET BALANCING

        def FlowRule(model, o, d, p, t):
            return sum(self.model.h_flow[(k, p, t)] for k in self.data.OD_PATHS[(o, d)]) >= self.data.D_DEMAND[
                (o, d, p, t)]
        # THIS SHOULD BE AN EQUALITY; BUT THEN THE PROBLEM GETS EASIER WITH A LARGER THAN OR EQUAL
        self.model.Flow = Constraint(self.data.ODPTS, rule=FlowRule)


        # EMISSIONS
        def emissions_rule(model, t):
            return (self.model.total_emissions[t] == sum(
                self.data.E_EMISSIONS[i,j,m,r,f, p, t] * self.model.x_flow[i,j,m,r,f, p, t] for p in self.data.P_PRODUCTS
                for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m]))
        self.model.Emissions = Constraint(self.data.TS, rule=emissions_rule) #removed self.data.T_TIME_PERIODS

        # Emission limit
        def EmissionCapRule(model, t):
            return self.model.total_emissions[t] <= self.data.CO2_CAP[t] + self.model.z_emission[t]
        self.model.EmissionCap = Constraint(self.data.TS, rule=EmissionCapRule)
        

        # CAPACITY constraints (compared to 2018) - TEMPORARY
        # the model quickly becomes infeasible when putting such constraints on the model. Should be tailor-made!

        # def CapacityConstraints(model, i,j,m,p,t):
        #    a = (i,j,m)
        #    return self.model.x_flow[a,p,t] <= self.data.buildchain2018[(i,j,m,p)]*2
        # self.model.CapacityConstr = Constraint(self.data.APT,rule=CapacityConstraints)

        
        #CAPACITY
        def CapacitatedFlowRule(model,i,j,m,r,ii,jj,mm,rr,t):
            e = (i,j,m,r)
            a = (ii,jj,mm,rr)
            return (sum(self.model.x_flow[a, f, p, t] for p in self.data.P_PRODUCTS for f in self.data.FM_FUEL[m]) <= 0.5*(self.data.Q_EDGE_BASE_RAIL[e] +
                   + self.data.Q_EDGE_RAIL[e] * sum(self.model.v_edge[e, tau] for tau in self.data.T_TIME_PERIODS if tau <= t)))
        self.model.CapacitatedFlow = Constraint(self.data.EAT_RAIL, rule = CapacitatedFlowRule)
        
        #Num expansions
        def ExpansionLimitRule(model,i,j,m,r):
            e = (i,j,m,r)
            return (sum(self.model.v_edge[(e,t)] for t in self.data.T_TIME_PERIODS) <= 1)
        self.model.ExpansionCap = Constraint(self.data.E_EDGES_RAIL, rule = ExpansionLimitRule)
        
        #Terminal capacity constraint. We keep the old notation here, so we can distinguish between OD and transfer, if they take up different capacity.
        def TerminalCapRule(model, i, c, m,t):
            return(sum(self.model.h_flow[k, p, t] for k in self.data.ORIGIN_PATHS[(i,m)] for p in self.data.PT[c]) + 
                   sum(self.model.h_flow[k, p, t] for k in self.data.DESTINATION_PATHS[(i,m)] for p in self.data.PT[c]) +
                   sum(self.model.h_flow[k,p,t] for k in self.data.TRANSFER_PATHS[(i,m)] for p in self.data.PT[c]) <= 
                   self.data.Q_NODE_BASE[i,c,m]+self.data.Q_NODE[i,c,m]*sum(self.model.w_node[i,c,m,tau] for tau in self.data.T_TIME_PERIODS if tau <= t))
        self.model.TerminalCap = Constraint(self.data.NCMT, rule = TerminalCapRule)
        
        #Num expansions of terminal NEW -- how many times you can perform a step-wise increase of the capacity
        def TerminalCapExpRule(model, i, c,m):
            return(sum(self.model.w_node[i,c,m,t] for t in self.data.T_TIME_PERIODS) <= 1) # THIS WAS AT 4 .... self.data.INV_NODE[i,m,c])
        self.model.TerminalCapExp = Constraint(self.data.NCM, rule = TerminalCapExpRule)

        #Charging / Filling
        def ChargingCapArcRule(model, i, j, m, r,f, t):
            e = (i, j, m, r)
            return (sum(self.model.x_flow[a,f,p, t] for p in self.data.P_PRODUCTS
                       for a in self.data.AE_ARCS[e]) <= self.data.Q_CHARGE_BASE[(e,f)] +
                   sum(self.model.y_charge[(e,f,tau)] for tau in self.data.T_TIME_PERIODS if tau <= t))
        self.model.ChargingCapArc = Constraint(self.data.EFT_CHARGE, rule=ChargingCapArcRule)
        #AIM also looked into charging infrastructure in NODES

        #Upgrading
        def InvestmentInfraRule(model,i,j,m,r,f,t):
            e = (i,j,m,r)
            return (sum(self.model.x_flow[a,f,p,t] for p in self.data.P_PRODUCTS for a in self.data.AE_ARCS[e])
                    <= self.data.BIG_M_UPG[e]*sum(self.model.u_upg[e,f,tau] for tau in self.data.T_TIME_PERIODS if tau <= t))
        self.model.InvestmentInfra = Constraint(self.data.UT_UPG, rule = InvestmentInfraRule)

        
        #def Diesel2020(model, t):
        #    return (sum(self.model.x_flow[(a,p,t)] for p in self.data.P_PRODUCTS
        #        for a in self.data.DIESEL_ROAD) >= sum(self.model.x_flow[(a,p,t)]
        #       for p in self.data.P_PRODUCTS for a in self.data.ARCS_ROAD))
        #self.model.Diesel2020Rate = Constraint(self.data.T_TIME_2020, rule=Diesel2020)
 
        # """
        # def NonAnticipativityRule(model,a,p):
        #     return(self.model.x_flow[(a, p, 2020, "average")] == self.model.x_flow[(a, p, 2020, "low")]
        #     == self.model.x_flow[(a, p, 2020, "high")] == self.model.x_flow[(a, p, 2020, "hydrogen")])
        #     self.model.NonAnticipativity = Constraint(self.data.AP, rule=NonAnticipativityRule)
        #     """
        
        if True:
        
            #Fleet Renewal
            def TotalTransportAmountRule(model,m,f,t):
                return (self.model.q_transp_amount[m,f,t] == sum( self.data.AVG_DISTANCE[a]*self.model.x_flow[a,f,p,t] for p in self.data.P_PRODUCTS 
                                                                for a in self.data.AM_ARCS[m]))
            self.model.TotalTranspAmount = Constraint(self.data.MFT, rule = TotalTransportAmountRule)
            
            def PositivePart1Rule(model,m,f,t):
                return (self.model.ppqq[m,f,t] >= self.model.q_transp_amount[m,f,t] - self.model.q_transp_amount[m,f,self.data.T_MIN1[t]])
            self.model.PositivePart1 = Constraint(self.data.MFT_MIN0, rule = PositivePart1Rule)
            
            def PositivePart2Rule(model,m,t):
                return (self.model.ppqq_sum[m,t] >= sum(self.model.q_transp_amount[m,f,t] - self.model.q_transp_amount[m,f,self.data.T_MIN1[t]] for f in self.data.FM_FUEL[m]))
            self.model.PositivePart2 = Constraint(self.data.MT_MIN0, rule = PositivePart2Rule)
            
            def FleetRenewalRule(model,m,t):
                return (sum(self.model.ppqq[m,f,t] for f in self.data.FM_FUEL[m]) <= self.data.RHO_FLEET_RENEWAL_RATE[m,t]*sum(
                        self.model.q_transp_amount[m,f,self.data.T_MIN1[t]]for f in self.data.FM_FUEL[m]) + self.model.ppqq_sum[m,t])
            self.model.FleetRenewal = Constraint(self.data.MT_MIN0, rule = FleetRenewalRule)
            
            #Technology maturity limit
            def TechMaturityLimitRule(model, m, f, t):
                return (self.model.q_transp_amount[(m,f,t)] <= self.data.Q_TECH[(m,f,t)])   #CHANGE THIS Q_TECH to mu*M
            self.model.TechMaturityLimit = Constraint(self.data.MFT_MATURITY, rule = TechMaturityLimitRule)

        return self.model
    

    # def update_tech_constraint(self):
    #     self.model.del_component(self.model.TechMaturityLimit);
    #     self.model.del_component(self.model.TechMaturityLimit_index);
    #     self.model.add_component("TechMaturityLimit",Constraint(self.data.MFTS_CAP, rule=self.TechMaturityLimitRule))    

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
        


        
