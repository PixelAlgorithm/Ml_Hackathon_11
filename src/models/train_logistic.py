"""Model 1: independent, reproducible logistic regression baseline."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,log_loss,precision_score,recall_score,accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.features.build_candidates import FEATURE_COLUMNS

SEED=20261118

def model():
    # Scaling is FIT ONLY on the training split/fold, never on student test.
    return make_pipeline(StandardScaler(),LogisticRegression(max_iter=250,class_weight='balanced',random_state=SEED))


def scores(m, data):
    y=data.label.astype(int).to_numpy()
    x=data[FEATURE_COLUMNS]
    p=m.predict_proba(x)[:,1]
    score_rows=data[['trip_id','obs_idx','arc_id','label']].copy()
    score_rows['probability']=p
    best=score_rows.sort_values('probability',ascending=False).drop_duplicates(['trip_id','obs_idx'])
    # If pseudo-label is not among candidates, top-1 automatically fails.
    reference_hits=int(best.label.sum())
    obs_total=best.shape[0]
    pred=(p>=0.5).astype(int)
    return {
        'candidate_macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'candidate_precision_positive':float(precision_score(y,pred,zero_division=0)),
        'candidate_recall_positive':float(recall_score(y,pred,zero_division=0)),
        'candidate_log_loss':float(log_loss(y,np.clip(p,1e-10,1-1e-10),labels=[0,1])),
        'observation_top1_accuracy_proxy':float(reference_hits/obs_total),
        'observations':int(obs_total),
        'candidate_rows':int(len(data)),
        'pseudo_label_candidate_coverage':float(data.groupby(['trip_id','obs_idx']).label.max().mean())
    }


def train(out):
    train_data=pd.read_csv(out/'train_candidates.csv',dtype={'trip_id':str})
    test_data=pd.read_csv(out/'student_test_candidates.csv',dtype={'trip_id':str})
    assert set(train_data.trip_id).isdisjoint(set(test_data.trip_id)), 'Training/dev trips overlap'
    x=train_data[FEATURE_COLUMNS];y=train_data.label.astype(int)
    groups=train_data.trip_id
    folds=GroupKFold(n_splits=3)
    cv=[]
    for fold,(tr,va) in enumerate(folds.split(x,y,groups),1):
        clf=model();start=time.perf_counter();clf.fit(x.iloc[tr],y.iloc[tr])
        result=scores(clf,train_data.iloc[va]);result['fold']=fold
        result['fit_seconds']=round(time.perf_counter()-start,2)
        cv.append(result)
        print('CV fold',fold,result,flush=True)
    clf=model();start=time.perf_counter();clf.fit(x,y)
    fit_s=time.perf_counter()-start
    start=time.perf_counter();results=scores(clf,test_data)
    inference_s=time.perf_counter()-start
    joblib.dump(clf,out/'logistic_baseline.joblib')
    report={'model':'LogisticRegression','seed':SEED,'hyperparameters':{'max_iter':250,'class_weight':'balanced'},
            'features':FEATURE_COLUMNS,'label_method':'NEAREST supplied .route arc (pseudo-label)',
            'cv_trip_grouped_folds':cv,'cv_mean_macro_f1':float(np.mean([v['candidate_macro_f1'] for v in cv])),
            'cv_mean_observation_top1_proxy':float(np.mean([v['observation_top1_accuracy_proxy'] for v in cv])),
            'student_test':results,'train_seconds':fit_s,'test_inference_and_scoring_seconds':inference_s}
    with open(out/'logistic_baseline_results.json','w') as f:json.dump(report,f,indent=2)
    print('STUDENT TEST:',results,flush=True)
    print('Saved:',out/'logistic_baseline.joblib',out/'logistic_baseline_results.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('results'))
    train(parser.parse_args().out)
