import copy
import numpy as np
import os
import sys

def interpolate(orig_data, time_periods, num_first_stage_periods):
    # copy the original data 
    # (we'll only change time-dependent data that should be interpolated)
    new_data = copy.deepcopy(orig_data) 

    # TIMING

    # update time periods
    new_data.T_TIME_PERIODS = time_periods

    # update first and second stage periods
    new_data.T_TIME_FIRST_STAGE_BASE = time_periods[0:num_first_stage_periods]
    new_data.T_TIME_SECOND_STAGE_BASE = time_periods[num_first_stage_periods:len(time_periods)]

    # define left and right endpoints for interpolation
    new_data.T_LEFT = {}
    new_data.T_RIGHT = {}
    for t in time_periods:
        # initialize
        new_data.T_LEFT[t] = -np.infty 
        new_data.T_RIGHT[t] = np.infty 
        # find correct value
        for tt in orig_data.T_TIME_PERIODS:
            if tt <= t:
                new_data.T_LEFT[t] = tt
        for tt in reversed(orig_data.T_TIME_PERIODS):
            if tt >= t:
                new_data.T_RIGHT[t] = tt

    LAST_PERIOD = orig_data.T_TIME_PERIODS[len(orig_data.T_TIME_PERIODS)-1]
    PENULT_PERIOD = orig_data.T_TIME_PERIODS[len(orig_data.T_TIME_PERIODS)-2]


    # update other sets based on the above
    new_data.T_MIN1 = {new_data.T_TIME_PERIODS[tt]:new_data.T_TIME_PERIODS[tt-1] for tt in range(1,len(new_data.T_TIME_PERIODS))} 
    new_data.T_TIME_PERIODS_ALL = new_data.T_TIME_PERIODS
    new_data.T_TIME_PERIODS_INIT = [new_data.T_TIME_PERIODS[0]]

    

    # DEMAND

    # demand
    new_data.D_DEMAND = {(o,d,p,t):0 for t in new_data.T_TIME_PERIODS for (o,d,p) in new_data.ODP} 
    for (o,d,p) in new_data.ODP:
        for t in new_data.T_TIME_PERIODS:
            if new_data.T_LEFT[t] == -np.infty:
                # extrapolate into the past
                raise Exception("We haven't implemented extrapolation into the past yet")
            elif new_data.T_RIGHT[t] == np.infty:
                # extrapolate into the future (linearly)
                last_value = orig_data.D_DEMAND[(o,d,p,LAST_PERIOD)]
                penult_value = orig_data.D_DEMAND[(o,d,p,PENULT_PERIOD)]
                new_data.D_DEMAND[(o,d,p,t)] = last_value + (t - LAST_PERIOD) * max(last_value - penult_value, 0) / (LAST_PERIOD - PENULT_PERIOD)
            else:
                # interpolate between left and right (linearly)
                t_left = new_data.T_LEFT[t]
                t_right = new_data.T_RIGHT[t]
                left_value = orig_data.D_DEMAND[(o,d,p,t_left)]
                right_value = orig_data.D_DEMAND[(o,d,p,t_right)]
                if t_left == t_right:
                    # we have exact data
                    new_data.D_DEMAND[(o,d,p,t)] = left_value
                else:
                    # interpolate
                    new_data.D_DEMAND[(o,d,p,t)] = left_value + (t - t_left) * (right_value - left_value) / (t_right - t_left)
            new_data.D_DEMAND[(o,d,p,t)] = round(new_data.D_DEMAND[(o,d,p,t)], new_data.precision_digits)
    # TODO: GET RID OF NEAR-ZERO DEMAND?

    # aggregate demand
    new_data.D_DEMAND_AGGR = {t:0 for t in new_data.T_TIME_PERIODS}
    for (o,d,p,t), value in new_data.D_DEMAND.items():
        new_data.D_DEMAND_AGGR[t] += value

    
    # OTHER PARAMETERS

    new_data.CO2_fee = {t: 10000000 for t in new_data.T_TIME_PERIODS}   #UNIT: nok/gCO2
    for t in new_data.T_TIME_PERIODS:
        if new_data.T_LEFT[t] == -np.infty:
            # extrapolate into the past
            raise Exception("We haven't implemented extrapolation into the past yet")
        elif new_data.T_RIGHT[t] == np.infty:
            # extrapolate into the future (linearly)
            last_value = orig_data.CO2_fee[LAST_PERIOD]
            penult_value = orig_data.CO2_fee[PENULT_PERIOD]
            new_data.CO2_fee[t] = last_value + (t - LAST_PERIOD) * max(last_value - penult_value, 0) / (LAST_PERIOD - PENULT_PERIOD)
        else:
            # interpolate between left and right (linearly)
            t_left = new_data.T_LEFT[t]
            t_right = new_data.T_RIGHT[t]
            left_value = orig_data.CO2_fee[t_left]
            right_value = orig_data.CO2_fee[t_right]
            if t_left == t_right:
                new_data.CO2_fee[t] = left_value
            else:
                new_data.CO2_fee[t] = left_value + (t - t_left) * (right_value - left_value) / (t_right - t_left)
        new_data.CO2_fee[t] = round(new_data.CO2_fee[t],new_data.precision_digits)

    #base level transport costs (in average scenario)
    new_data.C_TRANSP_COST_BASE = {(i,j,m,r,f,p,t): 1000000 for (i,j,m,r) in new_data.A_ARCS for f in new_data.FM_FUEL[m] 
                            for p in new_data.P_PRODUCTS for t in new_data.T_TIME_PERIODS}   #UNIT: NOK/T
    #scenario-dependent transport cost (computed using base cost)
    new_data.C_TRANSP_COST_NORMALIZED = {(m,f,p,t): 1000000 for m in new_data.M_MODES for f in new_data.FM_FUEL[m] 
                            for p in new_data.P_PRODUCTS for t in new_data.T_TIME_PERIODS}   #UNIT: NOK/Tkm
    
    new_data.E_EMISSIONS_NORMALIZED = {(m,f,p,t): 1000000 for m in new_data.M_MODES for f in new_data.FM_FUEL[m] 
                        for p in new_data.P_PRODUCTS for t in new_data.T_TIME_PERIODS}      #UNIT:  gCO2/T
    new_data.C_TRANSP_COST = {(i,j,m,r,f,p,t): 1000000 for (i,j,m,r) in new_data.A_ARCS for f in new_data.FM_FUEL[m] 
                            for p in new_data.P_PRODUCTS for t in new_data.T_TIME_PERIODS}   #UNIT: NOK/T
    new_data.E_EMISSIONS = {(i,j,m,r,f,p,t): 1000000 for (i,j,m,r) in new_data.A_ARCS for f in new_data.FM_FUEL[m] 
                        for p in new_data.P_PRODUCTS for t in new_data.T_TIME_PERIODS}      #UNIT:  gCO2/T
    new_data.C_CO2 = {(i,j,m,r,f,p,t): 1000000 for (i,j,m,r) in new_data.A_ARCS for f in new_data.FM_FUEL[m] 
                    for p in new_data.P_PRODUCTS for t in new_data.T_TIME_PERIODS}   #UNIT: nok/T

    # everything with index (m,f,p,t)
    # that is: C_TRANSP_COST_NORMALIZED, E_EMISSIONS_NORMALIZED
    for m in new_data.M_MODES:
        for f in new_data.FM_FUEL[m]:
            for p in new_data.P_PRODUCTS:
                for t in new_data.T_TIME_PERIODS:
                    if new_data.T_LEFT[t] == -np.infty:
                        # extrapolate into the past
                        raise Exception("We haven't implemented extrapolation into the past yet")
                    elif new_data.T_RIGHT[t] == np.infty:
                        # extrapolate into the future (linearly)
                        # C_TRANSP_COST_NORMALIZED
                        last_value = orig_data.C_TRANSP_COST_NORMALIZED[(m,f,p,LAST_PERIOD)]
                        penult_value = orig_data.C_TRANSP_COST_NORMALIZED[(m,f,p,PENULT_PERIOD)]
                        new_data.C_TRANSP_COST_NORMALIZED[(m,f,p,t)] = last_value + (t - LAST_PERIOD) * max(last_value - penult_value, 0) / (LAST_PERIOD - PENULT_PERIOD)
                        # E_EMISSIONS_NORMALIZED
                        last_value = orig_data.E_EMISSIONS_NORMALIZED[(m,f,p,LAST_PERIOD)]
                        penult_value = orig_data.E_EMISSIONS_NORMALIZED[(m,f,p,PENULT_PERIOD)]
                        new_data.E_EMISSIONS_NORMALIZED[(m,f,p,t)] = last_value + (t - LAST_PERIOD) * max(last_value - penult_value, 0) / (LAST_PERIOD - PENULT_PERIOD)
                    else:
                        # interpolate between left and right (linearly)
                        t_left = new_data.T_LEFT[t]
                        t_right = new_data.T_RIGHT[t]
                        # C_TRANSP_COST_NORMALIZED
                        left_value = orig_data.C_TRANSP_COST_NORMALIZED[(m,f,p,t_left)]
                        right_value = orig_data.C_TRANSP_COST_NORMALIZED[(m,f,p,t_right)]
                        if t_left == t_right:
                            new_data.C_TRANSP_COST_NORMALIZED[(m,f,p,t)] = left_value
                        else:
                            new_data.C_TRANSP_COST_NORMALIZED[(m,f,p,t)] = left_value + (t - t_left) * (right_value - left_value) / (t_right - t_left)
                        # E_EMISSIONS_NORMALIZED
                        left_value = orig_data.E_EMISSIONS_NORMALIZED[(m,f,p,t_left)]
                        right_value = orig_data.E_EMISSIONS_NORMALIZED[(m,f,p,t_right)]
                        if t_left == t_right:
                            new_data.E_EMISSIONS_NORMALIZED[(m,f,p,t)] = left_value
                        else:
                            new_data.E_EMISSIONS_NORMALIZED[(m,f,p,t)] = left_value + (t - t_left) * (right_value - left_value) / (t_right - t_left)
                    new_data.C_TRANSP_COST_NORMALIZED[(m,f,p,t)] = round(new_data.C_TRANSP_COST_NORMALIZED[(m,f,p,t)],new_data.precision_digits)
                    new_data.E_EMISSIONS_NORMALIZED[(m,f,p,t)] = round(new_data.E_EMISSIONS_NORMALIZED[(m,f,p,t)],new_data.precision_digits)
    # everything with index (i,j,m,r,f,p,t)
    # that is: C_TRANSP_COST_BASE, C_TRANSP_COST, E_EMISSIONS, C_CO2
    for (i,j,m,r) in new_data.A_ARCS:
        for f in new_data.FM_FUEL[m]:
            for p in new_data.P_PRODUCTS:
                for t in new_data.T_TIME_PERIODS:
                    if new_data.T_LEFT[t] == -np.infty:
                        # extrapolate into the past
                        raise Exception("We haven't implemented extrapolation into the past yet")
                    elif new_data.T_RIGHT[t] == np.infty:
                        # extrapolate into the future (linearly)
                        # C_TRANSP_COST_BASE
                        last_value = orig_data.C_TRANSP_COST_BASE[(i,j,m,r,f,p,LAST_PERIOD)]
                        penult_value = orig_data.C_TRANSP_COST_BASE[(i,j,m,r,f,p,PENULT_PERIOD)]
                        new_data.C_TRANSP_COST_BASE[(i,j,m,r,f,p,t)] = last_value + (t - LAST_PERIOD) * max(last_value - penult_value, 0) / (LAST_PERIOD - PENULT_PERIOD)
                        # C_TRANSP_COST
                        last_value = orig_data.C_TRANSP_COST[(i,j,m,r,f,p,LAST_PERIOD)]
                        penult_value = orig_data.C_TRANSP_COST[(i,j,m,r,f,p,PENULT_PERIOD)]
                        new_data.C_TRANSP_COST[(i,j,m,r,f,p,t)] = last_value + (t - LAST_PERIOD) * max(last_value - penult_value, 0) / (LAST_PERIOD - PENULT_PERIOD)
                        # E_EMISSIONS
                        last_value = orig_data.E_EMISSIONS[(i,j,m,r,f,p,LAST_PERIOD)]
                        penult_value = orig_data.E_EMISSIONS[(i,j,m,r,f,p,PENULT_PERIOD)]
                        new_data.E_EMISSIONS[(i,j,m,r,f,p,t)] = last_value + (t - LAST_PERIOD) * max(last_value - penult_value, 0) / (LAST_PERIOD - PENULT_PERIOD)
                        # C_CO2
                        last_value = orig_data.C_CO2[(i,j,m,r,f,p,LAST_PERIOD)]
                        penult_value = orig_data.C_CO2[(i,j,m,r,f,p,PENULT_PERIOD)]
                        new_data.C_CO2[(i,j,m,r,f,p,t)] = last_value + (t - LAST_PERIOD) * max(last_value - penult_value, 0) / (LAST_PERIOD - PENULT_PERIOD)
                    else:
                        # interpolate between left and right (linearly)
                        t_left = new_data.T_LEFT[t]
                        t_right = new_data.T_RIGHT[t]
                        # C_TRANSP_COST_BASE
                        left_value = orig_data.C_TRANSP_COST_BASE[(i,j,m,r,f,p,t_left)]
                        right_value = orig_data.C_TRANSP_COST_BASE[(i,j,m,r,f,p,t_right)]
                        if t_left == t_right:
                            new_data.C_TRANSP_COST_BASE[(i,j,m,r,f,p,t)] = left_value
                        else:
                            new_data.C_TRANSP_COST_BASE[(i,j,m,r,f,p,t)] = left_value + (t - t_left) * (right_value - left_value) / (t_right - t_left)
                        # C_TRANSP_COST
                        for s in orig_data.S_SCENARIOS:
                            left_value = orig_data.C_TRANSP_COST[(i,j,m,r,f,p,t_left,s)]
                            right_value = orig_data.C_TRANSP_COST[(i,j,m,r,f,p,t_right,s)]
                            if t_left == t_right:
                                new_data.C_TRANSP_COST[(i,j,m,r,f,p,t,s)] = left_value
                            else:
                                new_data.C_TRANSP_COST[(i,j,m,r,f,p,t,s)] = left_value + (t - t_left) * (right_value - left_value) / (t_right - t_left)
                        # E_EMISSIONS
                        left_value = orig_data.E_EMISSIONS[(i,j,m,r,f,p,t_left)]
                        right_value = orig_data.E_EMISSIONS[(i,j,m,r,f,p,t_right)]
                        if t_left == t_right:
                            new_data.E_EMISSIONS[(i,j,m,r,f,p,t)] = left_value
                        else:
                            new_data.E_EMISSIONS[(i,j,m,r,f,p,t)] = left_value + (t - t_left) * (right_value - left_value) / (t_right - t_left)
                        # C_CO2
                        left_value = orig_data.C_CO2[(i,j,m,r,f,p,t_left)]
                        right_value = orig_data.C_CO2[(i,j,m,r,f,p,t_right)]
                        if t_left == t_right:
                            new_data.C_CO2[(i,j,m,r,f,p,t)] = left_value
                        else:
                            new_data.C_CO2[(i,j,m,r,f,p,t)] = left_value + (t - t_left) * (right_value - left_value) / (t_right - t_left)
                    new_data.C_TRANSP_COST_BASE[(i,j,m,r,f,p,t)] = round(new_data.C_TRANSP_COST_BASE[(i,j,m,r,f,p,t)],new_data.precision_digits)
                    new_data.C_TRANSP_COST[(i,j,m,r,f,p,t,s)] = round(new_data.C_TRANSP_COST[(i,j,m,r,f,p,t,s)],new_data.precision_digits)
                    new_data.E_EMISSIONS[(i,j,m,r,f,p,t)] = round(new_data.E_EMISSIONS[(i,j,m,r,f,p,t)],new_data.precision_digits)
                    new_data.C_CO2[(i,j,m,r,f,p,t)] = round(new_data.C_CO2[(i,j,m,r,f,p,t)] ,new_data.precision_digits)
    # initialize R_TECH_READINESS_MATURITY at base path
    for s in new_data.S_SCENARIOS:
        for (m,f) in new_data.tech_is_mature:
            if new_data.tech_is_mature[(m,f)]:
                for t in new_data.T_TIME_PERIODS:    
                    new_data.R_TECH_READINESS_MATURITY[(m, f, t,s)] = 100 # assumption: all mature technologies have 100% market potential
            else:
                for t in new_data.T_TIME_PERIODS:
                    new_data.R_TECH_READINESS_MATURITY[(m, f, t,s)] = new_data.tech_base_bass_model[(m,f)].A(t) # compute maturity level based on base Bass diffusion model 


    #Initializing transport work share in base year
    new_data.Q_SHARE_INIT_MAX = {}
    new_data.MFT_INIT_TRANSP_SHARE = []
    for m in new_data.M_MODES:
        for f in new_data.FM_FUEL[m]:
            new_data.Q_SHARE_INIT_MAX[(m,f,new_data.T_TIME_PERIODS[0])] = orig_data.Q_SHARE_INIT_MAX[(m,f,orig_data.T_TIME_PERIODS[0])]
            new_data.Q_SHARE_INIT_MAX[(m,f,new_data.T_TIME_PERIODS[0])] = orig_data.Q_SHARE_INIT_MAX[(m,f,orig_data.T_TIME_PERIODS[0])]
            new_data.MFT_INIT_TRANSP_SHARE.append((m,f,new_data.T_TIME_PERIODS[0]))


    # TODO: SOME ROUNDING?


    # UPDATE COMBINED SETS
    new_data.combined_sets()

    return new_data





# TESTING

if False:
    #os.chdir('M:/Documents/GitHub/AIM_Norwegian_Freight_Model') #uncomment this for stand-alone testing of this fille
    os.chdir('C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\GitHub\\AIM_Norwegian_Freight_Model\\AIM_Norwegian_Freight_Model')
    sys.path.insert(0, '') #make sure the modules are found in the new working directory
    from ConstructData import *

    # load original data (based on [2022, 2026, 2030, 2040, 2050])
    orig_data = TransportSets(sheet_name_scenarios='four_scenarios')

    # define new timeline
    time_periods = [2023, 2028, 2034, 2040, 2050]   # new time periods
    num_first_stage_periods = 2                                 # how many of the periods above are in first stage

    # define new data based on new timeline by interpolating between time periods in orig_data
    new_data = interpolate(orig_data, time_periods, num_first_stage_periods)



    for i in new_data.E_EMISSIONS:
        print(i, ": ", new_data.E_EMISSIONS[i])


















