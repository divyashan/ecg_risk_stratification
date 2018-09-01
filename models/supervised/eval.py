import numpy as np
import pandas as pd
import pdb

from lifelines import CoxPHFitter
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat

def risk_scores(model, patients, threshold=.5):
    scores = []
    for patient_beats in patients: 
        aa = np.expand_dims(patient_beats[:,2:], 2)
        output = model.predict(aa)
        pred_y = [1 if v > threshold else 0 for v in output]
        pid_score = np.mean(pred_y)
        #pid_score = np.mean(output)
        scores.append(pid_score)
    return scores 

def evaluate_AUC(scores, patient_labels):
    auc_val = roc_auc_score(patient_labels, scores)
    return auc_val

def evaluate_HR(scores, patients, patient_labels):
    outcome_mat = loadmat("./datasets/patient_outcomes.mat")['outcomes']
    survival_dict = {x[0]: x[4] for x in outcome_mat}
    df_list = []
    for risk_score, patient, outcome in zip(scores, patients, patient_labels):
        pid = patient[0][0]
        death_date = survival_dict[pid]
        patient_dict = {'risk': risk_score, 'pid': pid, 
                        'death': outcome, 'days_survived': death_date}
        df_list.append(patient_dict)
    patient_df = pd.DataFrame(df_list)
    jj = patient_df.isnull().sum().sum()
    cph = CoxPHFitter()
    m = cph.fit(patient_df, duration_col='days_survived', event_col='death')
    
    return np.exp(m.hazards_['risk'][0])


    
