import numpy as np
import pandas as pd
import pdb
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt    
from lifelines import CoxPHFitter
from lifelines import NelsonAalenFitter
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy.io import loadmat

def risk_scores(model, patients, threshold=.5):
    scores = []
    for patient_beats in patients: 
        output = model.predict(patient_beats)
        pred_y = [1 if v > threshold else 0 for v in output]
        pid_score = np.mean(pred_y)
        #pid_score = np.mean(output)
        scores.append(pid_score)
    return scores 

def evaluate_AUC(scores, patient_labels):
    auc_val = roc_auc_score(patient_labels, scores)
    return auc_val

def plot_HR(df, with_ci=False):
    T = df['days_survived']
    E  = df['death']
    naf = NelsonAalenFitter()

    cutoff = np.percentile(df['risk'], 75)
    high_risk = df['risk'] > cutoff
    
    naf.fit(T[high_risk], event_observed=E[high_risk], label='High_Risk')
    ax = naf.plot(ci_show=with_ci)
    naf.fit(T[~high_risk], event_observed=E[~high_risk], label='Low_Risk')
    naf.plot(ax=ax,  ci_show=with_ci)
    
    plt.ylim(0, .1);
    plt.xlabel("Days")
    plt.ylabel("Risk of Death")
    plt.title("Cardiovascular Death Risk over time (top quartile)")
    if with_ci:
        plt.savefig("./hr_with_ci.png")
    else:
        plt.savefig("./hr_without_ci.png")

def plot_AUC(scores, patient_labels):
    fpr, tpr, threshold =  roc_curve(patient_labels, scores)
    roc_auc =  auc(fpr, tpr)
    
    plt.title('ROC (Receiver Operating Characteristic Curve)', fontdict={'fontsize': 'x-large'})
    plt.plot(fpr, tpr, 'orange', label = 'ROC Curve (AUC = %0.2f)' % roc_auc)
    plt.legend(loc = 'lower right', fontsize='large', frameon=False)
    plt.plot([0, 1], [0, 1],'b--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./roc_auc_curve.png')

def evaluate_HR(survival_dict, scores, pids, patient_labels, hr_days=90, mode="continuous"):
    df_list = []
    for risk_score, pid, outcome in zip(scores, pids, patient_labels):
        death_date = survival_dict[pid]
        patient_dict = {'risk': risk_score, 'pid': pid, 
                        'death': outcome, 'days_survived': death_date}
        df_list.append(patient_dict)
    patient_df = pd.DataFrame(df_list)
    hr_days_opts = [90]
    hr_vals = []
    for hr_days in hr_days_opts:
	    patient_df.loc[patient_df['days_survived'] > hr_days, 'death'] = 0
	    patient_df.loc[patient_df['death'] == 0, 'days_survived'] = 100
            cph = CoxPHFitter()
	    m = cph.fit(patient_df, duration_col='days_survived', event_col='death')
   	    hr_vals.append(np.exp(m.hazards_['risk'][0])) 
    return hr_vals


    
