
# ------------------------------ IMPORT LIBRARIES --------------------------------- #

import pandas as pd
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier
from fastFM import als
import scipy.sparse as sp
from sklearn.externals import joblib
from sklearn import svm
import os


# ------------------------------ FIT MODELS --------------------------------- #

def validate_log_reg(params, X_train, y_train, X_valid, y_valid):
    """
    validate parameter for logistic regression
    @params[dict]
    """
    
    # create model object with given params
    model = LogisticRegression(C=params['C'],
                              penalty=params['penalty'],
                              class_weight=params['class_weight'],
                              tol=params['tol'])
    
    # fit model with data
    model = model.fit(X_train, y_train)
    
    # evaluate model with auc
    pred_valid = model.predict_proba(X_valid)[:, 1]
    auc_out = roc_auc_score(y_valid, pred_valid)
    
    return auc_out, model



def optimize_log_reg(params, 
                        X_train, y_train, 
                        X_valid, y_valid, 
                        save_model = False,
                        file_name='lg_dsp_002'):
    """
    optimize logistic regression model with different parms, 
    return model with best performance in validation dataset
    Input:
    @params[dict]: {k1: [v1, v2], ...}
    """
    
    print('start searching for optimal params...')
    best_model = None
    best_score = 0.0
    
    # search grid of parameters
    for param in ParameterGrid(params):
        # validate each parameter 
        score, model = validate_log_reg(param, X_train, y_train, X_valid, y_valid)
        if score > best_score:
            best_score = score
            best_model = model
            
    # save model into file        
    if save_model:
        joblib.dump(best_model_lg, file_name)
        print('saving the model to {}'.format(file_name))

    return best_model, best_score



def eval_ensemble_weight(a_pcts, pred_a, pred_b, y_valid):
    """
    evaluate different optimal ensemble weight by roc-auc
    """
    auc_esbs = []
    for a_pct in a_pcts:
        pred_esb = a_pct * pred_a + (1-a_pct) * pred_b
        auc_esb = roc_auc_score(y_valid, pred_esb)
        auc_esbs.append(auc_esb)
        
    return auc_esbs
                

# ------------------------------ Ensemble Model --------------------------------- #
def ensemble_gbm_lg(gbm, lg, X_valid_t, X_valid, gbm_pct=0.8):
    """
    Ensemble Gradient Boosting Tree base model and Logistic Regression model
    @X_valid_t[df]: features for gbm model
    @X_valid: features for lg model
    """
    
    pred_gbm = gbm.predict(X_valid_t)
    pred_lg  = lg.predict_proba(X_valid)[:,1]
    pred_esb = gbm_pct*pred_gbm + (1-gbm_pct)*pred_lg
    
    return pred_esb
    
    
    