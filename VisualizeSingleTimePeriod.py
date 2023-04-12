from Data.settings import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

import pickle
import json

#---------------------------------------------------------#
#       User Settings
#---------------------------------------------------------#

scenarios = "9Scen"   # '4Scen', '9Scen','AllScen'
analysis = "SP"
years = [2034,2050]


#---------------------------------------------------------#
#       Output data
#---------------------------------------------------------#

for year in years:

    with open(r'Data//output//'+analysis+'_'+scenarios+'.pickle', 'rb') as output_file:
        output = pickle.load(output_file)

    with open(r'Data//output//'+analysis+'_'+scenarios+'_single_time_period_'+str(year)+'.pickle', 'rb') as output_file:
        output_ltp = pickle.load(output_file)

    with open(r'Data//base_data//'+scenarios+'.pickle', 'rb') as data_file:
        base_data = pickle.load(data_file)

    year_index = base_data.T_TIME_PERIODS.index(year)
    run_identifier = analysis+'_'+scenarios

    #---------------------------------------------------------#
    #       plot
    #---------------------------------------------------------#
    
    def mode_mix_calculations(output,base_data):
        output.x_flow['Distance'] = 0 # in Tonnes KM
        for index, row in output.x_flow.iterrows():
            output.x_flow.at[index,'Distance'] = base_data.AVG_DISTANCE[(row['from'],row['to'],row['mode'],row['route'])]

        output.x_flow['TransportArbeid'] = output.x_flow['Distance']*output.x_flow['weight'] /10**9*SCALING_FACTOR_WEIGHT # in GTonnes KM

        #for i in [1]:  #we only do one type, discuss what foreign transport needs to be excluded
        i = 1
        ss_x_flow = output.x_flow
        TranspArb = ss_x_flow[['mode','fuel','time_period','TransportArbeid','scenario']].groupby(['mode','fuel','time_period','scenario'], as_index=False).agg({'TransportArbeid':'sum'})
                
        TotalTranspArb = TranspArb.groupby(['time_period','scenario'], as_index=False).agg({'TransportArbeid':'sum'})
        TotalTranspArb = TotalTranspArb.rename(columns={"TransportArbeid": "TransportArbeidTotal"})
        TranspArb = TranspArb.rename(columns={"TransportArbeid": "TranspArb"})

        TranspArb = pd.merge(TranspArb,TotalTranspArb,how='left',on=['time_period','scenario'])
        TranspArb['RelTranspArb'] = 100*TranspArb['TranspArb'] / TranspArb['TransportArbeidTotal']
        
        TranspArb['RelTranspArb_std'] = TranspArb['RelTranspArb']
        TranspArb['TranspArb_std'] = TranspArb['TranspArb']
        MFTS = [(m,f,t,s) for (m,f,t) in base_data.MFT for s in base_data.S_SCENARIOS]
        all_rows = pd.DataFrame(MFTS, columns = ['mode', 'fuel', 'time_period','scenario'])
        TranspArb = pd.merge(all_rows,TranspArb,how='left',on=['mode','fuel','time_period','scenario']).fillna(0)

        TranspArbAvgScen = TranspArb[['mode','fuel','time_period','scenario','TranspArb','TranspArb_std','RelTranspArb','RelTranspArb_std']].groupby(
            ['mode','fuel','time_period'], as_index=False).agg({'TranspArb':'mean','TranspArb_std':'std','RelTranspArb':'mean','RelTranspArb_std':'std'})
        TranspArbAvgScen = TranspArbAvgScen.fillna(0) #in case of a single scenario we get NA's

        return output, TranspArbAvgScen

    output, TranspArbAvgScen = mode_mix_calculations(output,base_data)
    output_ltp, TranspArbAvgScen_ltp = mode_mix_calculations(output_ltp,base_data)

    def plot_mode_mixes(TranspArbAvgScen,TranspArbAvgScen_ltp, base_data,absolute_transp_work=True):  #result data = TranspArbAvgScen
        
        #https://matplotlib.org/stable/gallery/color/named_colors.html
        color_dict = {'Diesel':                 'firebrick', 
                        'Ammonia':              'royalblue', 
                        'Hydrogen':             'deepskyblue', 
                        'Battery electric':     'mediumseagreen',
                        'Battery train':        'darkolivegreen', 
                        'Electric train (CL)':  'mediumseagreen', 
                        'LNG':                  'blue', 
                        'MGO':                  'darkviolet', 
                        'Biogas':               'teal', 
                        'Biodiesel':            'darkorange', 
                        'Biodiesel (HVO)':      'darkorange', 
                        'HFO':                  'firebrick'           }

        labels = ['base ', 'static '] # +str(year)
        width = 0.7       # the width of the bars: can also be len(x) sequence

        base_string = 'TranspArb'
        ylabel = 'Transport work (GTonnes-kilometer)'
        if not absolute_transp_work:
            base_string = 'Rel'+base_string
            ylabel = 'Relative transport work (%)'

        ymax = {"Road":45, "Rail":6, "Sea":110}
        for m in ["Road", "Rail", "Sea"]:

            fig, ax = plt.subplots(figsize=(1.8,5))  #

            leftright = -0.3
            bottom = [0 for i in range(2)]  
            for f in base_data.FM_FUEL[m]:
                subset = TranspArbAvgScen[(TranspArbAvgScen['mode']==m)&(TranspArbAvgScen['fuel']==f)]
                subset_ltp = TranspArbAvgScen_ltp[(TranspArbAvgScen_ltp['mode']==m)&(TranspArbAvgScen_ltp['fuel']==f)]

                yval = subset[base_string].tolist()[year_index]
                yval_ltp = subset_ltp[base_string].tolist()[year_index]
                yvals = [yval,yval_ltp]

                yerror = subset[base_string+'_std'].tolist()
                yerror_ltp = subset_ltp[base_string+'_std'].tolist()
                yerr = [yerror[year_index],yerror_ltp[year_index]]
                
                trans1 = Affine2D().translate(leftright, 0.0) + ax.transData

                do_not_plot_lims = [False]*len(yerr)
                for i in range(len(yerr)):
                    if yerr[i] > 0.001:
                        do_not_plot_lims[i] = True

                n_start = 5
                for i in range(len(yerr)):
                    if yerr[i] > 0.001:
                        n_start = i
                        break
                #this works! (preventing standard deviations to be plotted, when they are not there)

                ax.bar(labels, yvals, 
                            width, 
                            yerr=yerr, 
                            bottom = bottom,
                            error_kw={#'elinewidth':6,'capsize':6,'capthick':6,'ecolor':'black',
                                'capsize':2,
                                'errorevery':(n_start,1),'transform':trans1, 
                                #'xlolims':do_not_plot_lims,'xuplims':do_not_plot_lims
                                },   #elinewidth, capthickfloat
                            label=f,
                            color=color_dict[f],)
                bottom = [bottom[i] + yvals[i] for i in range(len(bottom))]
                leftright = leftright + 0.09
            ax.set_ylabel(ylabel)
            #ax.set_title(m + ' - ' + analysis_type)
            
            ax.axis(ymin=0,
                    ymax=ymax[m]
                    #ymax=ax.get_ylim()[1]
                    )
            #ax.legend() #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #correct

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            #ax.axvline(x = 1.5, color = 'black',ls='--') 
            #ax.text(0.5, 0.95*ax.get_ylim()[1], "First stage", fontdict=None)
            #ax.text(1.6, 0.95*ax.get_ylim()[1], "Second stage", fontdict=None)
            fig.tight_layout()
            fig.savefig(r"Data//Figures//"+run_identifier+"_single_time_period_"+str(year)+'_'+m+".png",dpi=300,bbox_inches='tight')
        

    plot_mode_mixes(TranspArbAvgScen,TranspArbAvgScen_ltp,base_data,absolute_transp_work=True)
    
#import os
#print(os.getcwd())

