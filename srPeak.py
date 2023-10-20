import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.tree import DecisionTreeRegressor
from matplotlib.backends.backend_pdf import PdfPages

from sklearn import tree

import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------------------------------------------------------------------

def tree_to_nodes(period, site_response, cp_alpha):
    """
    Sub-function to fit tree-regression and extract nodes from results for peak identification code.
    
    Input Arguments:
        period = array of periods
        site_response = array of residual site response, at periods specified in "period"
        cp_alpha = complexity parameter
        
    Output Arguments:
        nodes = array of period-values corresponding to tree-regression nodes
        values = array of site response-values corresponding to tree regression nodes
    
    """
    regressor = DecisionTreeRegressor(ccp_alpha=cp_alpha)
    regressor.fit(period.reshape(-1, 1), site_response.reshape(-1,1))
    x_freqlog = np.logspace(np.log10(min(period)),np.log10(max(period)),300)
    pred = regressor.predict(x_freqlog.reshape(-1, 1))
    pred_freq = np.exp(x_freqlog)
    y_val, unique_idx = np.unique(pred, return_index = True)
    x_interval = x_freqlog[np.sort(unique_idx)]
    nodes = np.append(x_interval, np.max(x_freqlog))
    values = pred[np.sort(unique_idx)]
    return(nodes, values)

def Identify_Site_Response_Peaks(period, site_response, standard_error, 
                                 cp_alpha = 0.0003, step_thres = 0.65, amp_thres = 0.27, 
                                 wid_thres = 2.3, k_thres = 0.9, plot = True):
    """
    Function to identify peak features in residual site response, using automated algorithm presented in Buckreis et al. (202x)
    
    Input Arguments:
        period = array of periods
        site_response = array of residual site response, at periods specified in "period"
        standard_error = array of standard errors for residual site response, at periods specified in "period"
        cp_alpha = complexity parameter (default = 0.0003)
        step_thres = threshold for width of "stable peak tails" (default = 0.65)
        amp_thres = threshold for relative peak amplitude = peak - tail (default = 0.27)
        wid_thres = threshold for width of peak (default = 2.3)
        k_thres = scaling constant for evaluating uncertainty of peak and tail amplitudes (default = 0.9)
        plot = option to plot the results or not (default = True)
        
    Output Arguments:
        if plot == 0: [peak_indicator]
            peak_indicator = Flase if there is no peak; True if there is a peak
        if plot == 1: [peak_indicator, fig]
            peak_indicator = False if there is no peak; True if there is a peak
            fig = matplotlib Figure object
    """
    # Screen our NaN from data:
    for i in np.arange(len(site_response)-1,-1,-1):
        if np.isnan(site_response[i]):
            period = np.delete(period, i)
            site_response = np.delete(site_response, i)
            standard_error = np.delete(standard_error, i)
        
    # Fit the Decision Tree and Extract Steps:
    nodes, values = tree_to_nodes(period, site_response, cp_alpha)
    step_x1, step_x2, step_value = [], [], values
    for i in range(len(nodes)-1):
        step_x1.append(nodes[i])
        step_x2.append(nodes[i+1])
    
    # Identify potential peak steps:
    left_indicies, peak_indicies, right_indicies, criteria, peak_indicator = [], [], [], [], []
    for m in range(len(step_value)-2):
        flag = 0
        m += 1
        if (step_value[m-1] < step_value[m]) & (step_value[m] > step_value[m+1]):
            subcriteria = []
            # Identify left peak-adjacent plateau:
            l = 1
            while (m - l) > 0:
                if (step_value[m-l] > step_value[m-l+1]):
                    stepl = m - l + 1 
                    break
                else:
                    wid_ml = np.log(step_x2[m-l]) - np.log(step_x1[m-l])
                    if wid_ml <= step_thres:
                        l += 1
                    else:
                        stepl = m - l #+ 1 
                        break
            if (m - l) == 0:
                stepl = 0
               
            # Identify right peak-adjacenet plateau:
            r = 1
            while (m + r) < len(step_value)-1:
                if (step_value[m+r] < step_value[m+r+1]):
                    stepr = m + r  
                    break
                else:
                    wid_mr = np.log(step_x2[m+r]) - np.log(step_x1[m+r])
                    if wid_mr <= step_thres:
                        r += 1
                    else:
                        stepr = m + r  
                        break
            if (m + r) == len(step_value)-1:
                stepr = len(step_value)-1
            
            # Check that the peak amplitude is greater than amp_thres:
            ampl = step_value[stepl]
            ampr = step_value[stepr]
            #print('amp_thres = %1.2f ?<? %1.4f'%(amp_thres, step_value[m] - max(ampl, ampr)))
            if step_value[m] - max(ampl, ampr) < amp_thres:
                subcriteria.append(0)
            else:
                subcriteria.append(1)
                
            # Check that the peak width is not too large:
            Tl = step_x2[stepl]
            Tr = step_x1[stepr]
            widp = np.log(Tr) - np.log(Tl)
            #print('wid_thres = %1.2f ?>? %1.4f'%(wid_thres, np.log(Tr) - np.log(Tl)))
            if widp > wid_thres:
                subcriteria.append(0)
            else:
                subcriteria.append(1)

                
            # Check that the uncertainty of the peak and peak-adjacent plateau amplitudes are acceptable:
            df = pd.DataFrame({'period':period, 'site_response':site_response, 'standard_error':standard_error})
            dfl = df.loc[(df.period >= step_x1[stepl]) & (df.period <= step_x2[stepl])]
            etal = np.mean(df['site_response'])
            SEl = np.mean(df['standard_error'])
            kl = (step_value[m] - etal)/SEl
            dfr = df.loc[(df.period >= step_x1[stepr]) & (df.period <= step_x2[stepr])]
            etar = np.mean(df['site_response'])
            SEr = np.mean(df['standard_error'])
            kr = (step_value[m] - etar)/SEr
            #print('k_thres = %1.2f ?<? %1.4f'%(k_thres, min(kl,kr)))
            if min(kl, kr) < k_thres:
                subcriteria.append(0)
            else:
                subcriteria.append(1)
            
            criteria.append(subcriteria)
            left_indicies.append(stepl)
            peak_indicies.append(m)
            right_indicies.append(stepr)
            if sum(subcriteria) == 3:
                # If the code makes it this far, then the potential peak step IS a peak step:
                #print('Step %i is a peak step.'%m)
                peak_indicator.append(True)
            else:
                peak_indicator.append(False)
    
    if plot == True:
    
        fig, (ax) = plt.subplots(1,1,dpi=100,figsize=(5,4))
        ax.set_xscale('log'); ax.set_xlabel('Period, $T$ (s)')
        ax.set_ylabel('Residual Site Response')
        ax.axhline(y=0,linewidth=0.5,color='gray',zorder=1)
        ax.fill_between(x=period, y1=list(site_response-standard_error), y2=list(site_response+standard_error), color='k', alpha=0.1, zorder=2)
        ax.plot(period, site_response, 'k-', linewidth=1, zorder=3)

        x = []; y = []
        for i in range(len(values)):
            x += [nodes[i], nodes[i+1]]
            y += [values[i], values[i]]
        plt.plot(x, y, '-', color='tab:red', linewidth=1, zorder=6)
        
        # Plot Identified peaks:
        try:
            df = pd.DataFrame({'x1':step_x1, 'x2':step_x2, 'value':step_value})
            for l, m, r in zip(left_indicies, peak_indicies, right_indicies):
                #ax.plot([step_x1[m],step_x2[m]],[step_value[m],step_value[m]], '-', color='gold', linewidth=2, zorder=5)
                df2 = df.loc[(df['x1'] >= step_x1[l]) & (df['x2'] <= step_x2[r])]
                xx, yy = [], []
                for i in df2.index:
                    xx.append(df2['x1'][i])
                    xx.append(df2['x2'][i])
                    yy.append(df2['value'][i])
                    yy.append(df2['value'][i])
                ax.plot(xx,yy, '-', color='tab:green', linewidth=3, zorder=4)
        except:
            pass
        
#         props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#         ax.text(0.015, 0.98, '%i usable records'%num_records, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', bbox=props)
        
        ax.set_ylim(min(ax.get_ylim()[0],-1.25), max(ax.get_ylim()[1],1.25))
        ax.set_xticks([0.001,0.01,0.1,1,10,100])
        ax.set_xticklabels([0.001,0.01,0.1,1,10,100])
        #ax.set_xlim(min(period),max(period))
        ax.set_xlim(0.01,10)
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        plt.close()
        
        return(peak_indicator, fig)
    
    DATA = [period, site_response, standard_error, 
            nodes, values, step_x1, step_x2, step_value, 
            left_indicies, peak_indicies, right_indicies, criteria,peak_indicator]
        
    return(peak_indicator, DATA)