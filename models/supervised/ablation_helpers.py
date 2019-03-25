import numpy as np
import pandas as pd
import pdb

from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings("error")

fig_dir = "/home/divyas/ecg_AAAI/models/supervised/figs"

def get_fig_path(y_mode, day_thresh, split_num, model_name):
    return fig_dir + "/" + y_mode + "/" + str(day_thresh) + "/split_" + split_num + "/" + model_name

def get_split_path(y_mode, day_thresh, split_num):
    return fig_dir + "/" + y_mode + "/" + str(day_thresh) + "/split_" + split_num


def get_day_path(y_mode, day_thresh):
    return fig_dir + "/" + y_mode + "/" + str(day_thresh)

def get_labels(fhandle, y_mode, day_thresh):
    y = fhandle[y_mode + "_labels"][:]
    y = np.array([1 if (y_val < day_thresh and y_val > 0) else 0 for y_val in y])
    return y

def top_10_mean(pred_vals):
    thresh = np.percentile(pred_vals, 90)
    filtered = [x for x in pred_vals if x >= thresh]
    if len(filtered) == 0:
        return np.mean(pred_vals)
    return np.mean(filtered)

def top_20_mean(pred_vals):
    thresh = np.percentile(pred_vals, 80)
    filtered = [x for x in pred_vals if x >= thresh]
    if len(filtered) == 0:
        return np.mean(pred_vals)
    return np.mean(filtered)

def reshape_X(X):
    reshaped_X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
    reshaped_X = np.expand_dims(reshaped_X, 2)
    return reshaped_X

def get_block_batch(x_train, y_train, bs, i, n_beats=3600):
    start = bs*i
    end = start + bs
    all_idxs = np.array(range(len(y_train)))
    sel_idxs = sorted(all_idxs.take(range(start, end), mode="wrap"))

    x_batch = x_train[sel_idxs]
    x_batch = reshape_X(x_batch)

    y_batch = y_train[sel_idxs]
    y_batch = np.array([[y_val]*n_beats for y_val in y_batch]).flatten()
    return x_batch, y_batch

def get_block_batch_deepsets(x_train, y_train, bs, i, n_beats=3600):
    start = bs*i
    end = start + bs
    all_idxs = np.array(range(len(y_train)))
    sel_idxs = sorted(all_idxs.take(range(start, end), mode="wrap"))
    
    x_batch = x_train[sel_idxs]
    y_batch = y_train[sel_idxs]
    #y_batch = np.array([[y_val]*n_beats for y_val in y_batch]).flatten()
    return x_batch, y_batch
    
def get_block(fhandle, i, block_size, y_mode, day_thresh, n_beats=3600):
    start = i*block_size
    end = start + block_size
    x_block = fhandle['adjacent_beats'][start:end,:n_beats]
    y_block = thresh_labels(fhandle[y_mode + "_labels"][start:end], day_thresh)
    return np.array(x_block), y_block

def get_labels(fhandle, y_mode, day_thresh):
    y = fhandle[y_mode + "_labels"][:]
    return thresh_labels(y, day_thresh)

def thresh_labels(y, day_thresh):
    thresh_y = np.array([1 if (y_val < day_thresh and y_val > 0) else 0 for y_val in y])
    return thresh_y

def calc_hr(true_y, pred_y, pctl=75, discretize=False):
    thresh = np.percentile(pred_y, pctl)
    dicts = []
    for d, pred in zip(true_y, pred_y):
        o = 1 if d > 0 else 0
        r = 1 if pred >= thresh else 0
        if discretize==False:
            r = pred
        dicts.append({'duration': d, 'observed': o, 'risk': r})
    data = pd.DataFrame(dicts)
    cph = CoxPHFitter()
    cph.fit(data, duration_col='duration', event_col='observed', show_progress=True)
    try:
        return np.exp(cph.hazards_['risk'][0])
    except:
        return np.exp(cph.hazards_['risk'])

def get_preds(m, test_file, pred_f=np.mean, n_beats=3600):
    py_pred = []
    batch_size = 500
    test_beats = test_file['adjacent_beats']
    n_test = test_beats.shape[0]
    n_batches = int(n_test/batch_size + 1)
    
    for i in range(n_batches):
        start = i*batch_size
        end = start + batch_size
        x_test_batch = reshape_X(test_beats[start:end,:n_beats,:])
        y_preds = m.predict(x_test_batch)
        y_preds = y_preds.reshape((int(len(y_preds)/n_beats), n_beats))
        #iy_pred.extend(extension)
        for j in range(y_preds.shape[0]):
            patient_preds = y_preds[j]
            py_pred.append(pred_f(patient_preds))
    return  np.array(py_pred)

def get_preds_lr(m, test_file, pred_f=np.mean, n_beats=3600):
    py_pred = []
    batch_size = 500
    test_beats = test_file['adjacent_beats']
    n_test = test_beats.shape[0]
    n_batches = int(n_test/batch_size + 1)
    
    for i in range(n_batches):
        start = i*batch_size
        end = start + batch_size
        x_test_batch = reshape_X(test_beats[start:end,:n_beats,:])
        y_preds = m.predict_proba(x_test_batch[:,:,0])[:,1]
        y_preds = y_preds.reshape((int(len(y_preds)/n_beats), n_beats))
        #iy_pred.extend(extension)
        for j in range(y_preds.shape[0]):
            patient_preds = y_preds[j]
            py_pred.append(pred_f(patient_preds))
    return  np.array(py_pred)

def get_preds_deepsets(rho, test_file, n_beats=3600):
    py_pred = []
    x_test = test_file['adjacent_beats'][:,:n_beats]
    
    swapped = np.swapaxes(x_test, 0, 1)
    swapped = np.expand_dims(swapped, 3)
    rho_test_input = [x for x in swapped]
    
    return rho.predict(rho_test_input)
