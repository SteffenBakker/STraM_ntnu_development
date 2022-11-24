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

        self.results = None  # results is a structure filled out later in solve_model()
        self.status = None  # status is a string filled out later in solve_model()
        self.model = ConcreteModel()
        self.opt = pyomo.opt.SolverFactory('gurobi') #gurobi
        self.data = data

    def construct_model(self):
        
        #Significant speed improvements can be obtained using the LinearExpression object when there are long, dense, linear expressions.
        # USe linearexpressions: https://pyomo.readthedocs.io/en/stable/advanced_topics/linearexpression.html
        

        "VARIABLES"
        # Binary, NonNegativeReals, PositiveReals, etc

        self.model.x_flow = Var(self.data.AFPT, within=NonNegativeReals)
        self.model.b_flow = Var(self.data.AFVT, within=NonNegativeReals)
        self.model.h_flow = Var(self.data.KPT, within=NonNegativeReals)# flow on paths K,p
        self.model.h_flow_balancing = Var(self.data.KVT, within=NonNegativeReals)# flow on paths K,p

        self.model.StageCosts = Var(self.data.T_TIME_PERIODS, within = NonNegativeReals)
        
        self.model.epsilon_edge = Var(self.data.ET_RAIL, within = Binary) #within = Binary
        self.model.upsilon_upg = Var(self.data.UT_UPG, within = Binary) #bin.variable for investments upgrade/new infrastructure u at link l, time period t
        self.model.nu_node = Var(self.data.NCMT, within = Binary) #step-wise investment in terminals
        
        self.model.y_charge = Var(self.data.EFT_CHARGE, within=NonNegativeReals)
        self.model.z_emission = Var(self.data.TS, within = NonNegativeReals)

        self.model.total_emissions = Var(self.data.TS, within=NonNegativeReals) #instead of T_PERIODS!
                                        # bounds=emission_bound)  # just a variable to help with output
        
        self.model.q_transp_amount = Var(self.data.MFT, within=NonNegativeReals)
        self.model.q_max_transp_amount = Var(self.data.MFT, within=NonNegativeReals)

        "OBJECTIVE"
        #TO DO: check how the CO2 kicks in? Could be baked into C_TRANSP_COST (what is the question here?)
        def StageCostsVar(model, t):  
            # SOME QUICK TESTING SHOWED THAT SUM_PRODUCT IS QUITE A BIT SLOWER THAN SIMPLY TAKING THE SUM...
            yearly_transp_cost = (  sum((self.data.C_TRANSP_COST[(i,j,m,r,f,p,t)]+self.data.C_CO2[(i,j,m,r,f,p,t)])*self.model.x_flow[(i,j,m,r,f,p,t)] 
                                      for p in self.data.P_PRODUCTS for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m]) +
                                    sum( EMPTY_VEHICLE_FACTOR*(self.data.C_TRANSP_COST[(i,j,m,r,f,self.data.cheapest_product_per_vehicle[(m,f,t,v)],t)]+
                                          self.data.C_CO2[(i,j,m,r,f,self.data.cheapest_product_per_vehicle[(m,f,t,v)],t)]) * self.model.b_flow[(i,j,m,r,f,v,t)] 
                                          for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m] )
                                    #TO DO: take the minimum of transport costs and carbon costs across the product groups and multiply with 0.8 or so
                                )
            yearly_transfer_cost = sum(self.data.C_TRANSFER[(k,p)]*self.model.h_flow[k,p,t] for p in self.data.P_PRODUCTS  for k in self.data.MULTI_MODE_PATHS)
            delta = self.data.D_DISCOUNT_RATE**self.data.Y_YEARS[t][0]
            return(self.model.StageCosts[t] == (
                sum(self.data.D_DISCOUNT_RATE**n*(yearly_transp_cost+yearly_transfer_cost) for n in self.data.Y_YEARS[t]) + 
                sum(delta*self.data.C_EDGE_RAIL[e]*self.model.epsilon_edge[(e,t)] for e in self.data.E_EDGES_RAIL) +
                sum(delta*self.data.C_NODE[(i,c,m)]*self.model.nu_node[(i,c,m,t)] for (i, m) in self.data.NM_LIST_CAP for c in self.data.TERMINAL_TYPE[m] ) +
                # maybe N_NODES_CAP_NORWAY is needed?
                sum(delta*self.data.C_UPG[(e,f)]*self.model.upsilon_upg[(e,f,t)] for (e,f) in self.data.U_UPGRADE) +
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

        def PathArcRuleBalancing(model, i, j, m, r, v, t):
            a= (i,j,m,r)
            return sum(self.model.b_flow[a, f, v, t] for f in self.data.FM_FUEL[m]) == sum(
                self.model.h_flow_balancing[k, v, t] for k in self.data.KA_PATHS_UNIMODAL[a] )
        self.model.PathArcRelBalance = Constraint(self.data.AVT, rule=PathArcRuleBalancing)

        # FLEET BALANCING

        def FleetBalance(model, n,m,f,v, t):
            disbalance_in_node = (sum(self.model.x_flow[(a, f, p, t)] for a in self.data.ANM_ARCS_IN[(n,m)] for p in self.data.PV_PRODUCTS[v]) - 
                    sum(self.model.x_flow[(a, f, p, t)] for a in self.data.ANM_ARCS_OUT[(n,m)] for p in self.data.PV_PRODUCTS[v]))  
            empty_trips = (sum(self.model.b_flow[(a, f, v, t)] for a in self.data.ANM_ARCS_OUT[(n,m)]) -
                        sum(self.model.b_flow[(a, f, v, t)] for a in self.data.ANM_ARCS_IN[(n,m)]))            
            return (disbalance_in_node == empty_trips)
        # THIS SHOULD BE AN EQUALITY; BUT THEN THE PROBLEM GETS EASIER WITH A LARGER THAN OR EQUAL
        self.model.FleetBalance = Constraint(self.data.NMFVT, rule=FleetBalance)

        #-----------------------------------------------#

        # EMISSIONS
        def emissions_rule(model, t):
            return (
                self.model.total_emissions[t] == (sum(
                self.data.E_EMISSIONS[i,j,m,r,f, p, t] * self.model.x_flow[i,j,m,r,f, p, t] for p in self.data.P_PRODUCTS
                for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m]) + 
                sum(
                0 * self.model.b_flow[i,j,m,r,f, v, t] for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m]
                                                        for v in self.data.VEHICLE_TYPES_M[m]))
                )
                # TO DO: add emissions for empty trips (set at zero currently)
        
        self.model.Emissions = Constraint(self.data.TS, rule=emissions_rule) #removed self.data.T_TIME_PERIODS

        # Emission limit
        def EmissionCapRule(model, t):
            return self.model.total_emissions[t] <= self.data.CO2_CAP[t]/100*self.model.total_emissions[self.data.T_TIME_PERIODS[0]] + self.model.z_emission[t]
        self.model.EmissionCap = Constraint(self.data.TS, rule=EmissionCapRule)
        
        #-----------------------------------------------#

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
            return (sum(self.model.x_flow[a, f, p, t] for p in self.data.P_PRODUCTS for f in self.data.FM_FUEL[m]) + 
                    sum(self.model.b_flow[a, f, v, t] for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m] ) <= 0.5*(self.data.Q_EDGE_BASE_RAIL[e] +
                   + self.data.Q_EDGE_RAIL[e] * sum(self.model.epsilon_edge[e, tau] for tau in self.data.T_TIME_PERIODS if tau <= t)))
        self.model.CapacitatedFlow = Constraint(self.data.EAT_RAIL, rule = CapacitatedFlowRule)
        
        #Num expansions
        def ExpansionLimitRule(model,i,j,m,r):
            e = (i,j,m,r)
            return (sum(self.model.epsilon_edge[(e,t)] for t in self.data.T_TIME_PERIODS) <= 1)
        self.model.ExpansionCap = Constraint(self.data.E_EDGES_RAIL, rule = ExpansionLimitRule)
        
        #Terminal capacity constraint. We keep the old notation here, so we can distinguish between OD and transfer, if they take up different capacity.
        def TerminalCapRule(model, i, c, m,t):
            return(sum(self.model.h_flow[k, p, t] for k in self.data.ORIGIN_PATHS[(i,m)] for p in self.data.PT[c]) + 
                   sum(self.model.h_flow[k, p, t] for k in self.data.DESTINATION_PATHS[(i,m)] for p in self.data.PT[c]) +
                   sum(self.model.h_flow[k,p,t] for k in self.data.TRANSFER_PATHS[(i,m)] for p in self.data.PT[c]) <= 
                   self.data.Q_NODE_BASE[i,c,m]+self.data.Q_NODE[i,c,m]*sum(self.model.nu_node[i,c,m,tau] for tau in self.data.T_TIME_PERIODS if tau <= t))
        self.model.TerminalCap = Constraint(self.data.NCMT, rule = TerminalCapRule)
        
        #Num expansions of terminal NEW -- how many times you can perform a step-wise increase of the capacity
        def TerminalCapExpRule(model, i, c,m):
            return(sum(self.model.nu_node[i,c,m,t] for t in self.data.T_TIME_PERIODS) <= 1) # THIS WAS AT 4 .... self.data.INV_NODE[i,m,c])
        self.model.TerminalCapExp = Constraint(self.data.NCM, rule = TerminalCapExpRule)

        #Charging / Filling
        def ChargingCapArcRule(model, i, j, m, r,f, t):
            e = (i, j, m, r)
            return (sum(self.model.x_flow[a,f,p, t] for p in self.data.P_PRODUCTS
                       for a in self.data.AE_ARCS[e]) + sum(self.model.b_flow[a,f,v, t] for a in self.data.AE_ARCS[e] 
                        for v in self.data.VEHICLE_TYPES_M[m]) <= self.data.Q_CHARGE_BASE[(e,f)] +
                   sum(self.model.y_charge[(e,f,tau)] for tau in self.data.T_TIME_PERIODS if tau <= t))
        self.model.ChargingCapArc = Constraint(self.data.EFT_CHARGE, rule=ChargingCapArcRule)
        #AIM also looked into charging infrastructure in NODES

        #Upgrading
        def InvestmentInfraRule(model,i,j,m,r,f,t):
            e = (i,j,m,r)
            return (sum(self.model.x_flow[a,f,p,t] for p in self.data.P_PRODUCTS for a in self.data.AE_ARCS[e])
                    <= self.data.BIG_M_UPG[e]*sum(self.model.upsilon_upg[e,f,tau] for tau in self.data.T_TIME_PERIODS if tau <= t))
        self.model.InvestmentInfra = Constraint(self.data.UT_UPG, rule = InvestmentInfraRule)
    
        #-----------------------------------------------#
    
        #TransportArbeid
        def TotalTransportAmountRule(model,m,f,t):
            return (self.model.q_transp_amount[m,f,t] == sum( self.data.AVG_DISTANCE[a]*self.model.x_flow[a,f,p,t] for p in self.data.P_PRODUCTS 
                                                            for a in self.data.AM_ARCS[m]))
        self.model.TotalTranspAmount = Constraint(self.data.MFT, rule = TotalTransportAmountRule)
        
        #Technology maturity limit
        def TechMaturityLimitRule(model, m, f, t):
            return (self.model.q_transp_amount[(m,f,t)] <= self.data.R_TECH_READINESS_MATURITY[(m,f,t)]*sum(self.model.q_transp_amount[(m,ff,t)] for ff in self.data.FM_FUEL[m]))   #TO DO: CHANGE THIS Q_TECH to R*M
        self.model.TechMaturityLimit = Constraint(self.data.MFT_MATURITY, rule = TechMaturityLimitRule)

        #Max TransportArbeid
        def MaxTransportAmountRule(model,m,f,t,tau):
            return (self.model.q_max_transp_amount[m,f,t] >= self.model.q_transp_amount[m,f,tau])
        self.model.MaxTranspAmount = Constraint(self.data.MFTT, rule = MaxTransportAmountRule)
        
        #Fleet Renewal
        def FleetRenewalRule(model,m,f, t):
            decrease = self.model.q_transp_amount[(m,f,self.data.T_MIN1[t])] - self.model.q_transp_amount[(m,f,t)]
            factor = (t - self.data.T_MIN1[t]) / self.data.LIFETIME[(m,f)]
            return (decrease <= factor*self.model.q_max_transp_amount[m,f,t])
        self.model.FleetRenewal = Constraint(self.data.MFT_MIN0, rule = FleetRenewalRule)
        

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
                self.model.h_flow[(k,p,t)].fix(w)
            elif variable == 'epsilon_edge':
                self.model.epsilon_edge[(e,t)].fix(w)
            elif variable == 'upsilon_upg':
                self.model.upsilon_upg[(i,j,m,r,f,p,t)].fix(w)
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
                self.model.q_max_transp_amount[(m, f, t)].fix(w)


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
        


        
