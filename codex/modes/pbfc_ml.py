import os
import pandas as pd
import numpy as np
import sklearn.linear_model
import torch
from torch import nn
from torch.utils.data import DataLoader
import pbfc_data as data

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, precision_recall_fscore_support, make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

import shap
import json
import glob

import seaborn as sns
import matplotlib.pyplot as plt

from ..modules import output
'''
Prec, reacll, f1
'''
def metrics(Y_test:np.ndarray, Y_pred:np.ndarray):
    tp = 0
    fp = 0
    fn = 0

    for pred, actual in zip(Y_pred, Y_test):
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 1:
            fn += 1
    
    try:
        prec = tp/(tp+fp)
    except ZeroDivisionError:
        prec = 0
    try:
        rec = tp/(tp+fn)
    except ZeroDivisionError:
        rec = 0
    try:
        f1 = 2*prec*rec/(prec+rec)
    except ZeroDivisionError:
        f1 = 0

    return prec, rec, f1



def shap_calc(model, X_test, Y_test):
    plt.clf()
    class_names = Y_test.unique().tolist()
    feature_names = X_test.columns.tolist()
    print(class_names)
    print(feature_names)
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, plot_type='bar', class_names=class_names, feature_names=feature_names, show=True)
    plt.savefig('./bar.png')

    if len(class_names) == 2:
        plt.clf()
        shap.plots.violin(shap_values[0], X_test, plot_type='layered_violin', class_names=class_names, feature_names=feature_names, show=True)
        plt.savefig('./violin_bin.png')

    return shap_values


def ovr_ps_perf(model, X_test:pd.DataFrame, Y_test:pd.Series, experiment_name=None, model_name=None, orig_X_test=None, scaler=None, per_sample=True, output_dir='', notrain=False):
    perf_dict = {'model_name': model_name,
                       'split_id': experiment_name,
                       'test': {'Overall Performance': {},
                                'Per-Sample Performance': {}
                       }
                    }
    
    filename = f'perf_{experiment_name}_{model_name}.json'
    #filename = f'perf_{experiment_name}_{model_name}.json'
    '''if notrain:
        print("SKIPPING TEST:")
        return {}, filename'''
    
    X_test_data, Y_test_data, scaler_after = data.format_df_nd_all(X_test, Y_test, scaler)
    
    xtest_ids = X_test.index.tolist()
    ytest_ids = Y_test.index.tolist()
    assert xtest_ids == ytest_ids

    Y_pred = model.predict(X_test_data)  

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(model_name))
    print("MODEL: ", model_name)
    print("UNIQUE VAL YPRED", np.unique(Y_pred))
    print("UNIQUE VAL YTEST", np.unique(Y_test))
    assert(len(np.unique(Y_test))==len(np.unique(Y_pred))==2)

    avg_mode = 'binary'
    perf_dict['test']['Overall Performance']['accuracy'] = accuracy_score(Y_test_data, Y_pred)
    '''prec = precision_score(Y_test_data, Y_pred, average=avg_mode)
    rec = recall_score(Y_test_data, Y_pred, average=avg_mode)
    try:
        f1 = (2*prec*rec)/(prec+rec)
    except:
        f1 = 0.0'''
    prec, recall, f1 = metrics(Y_test, Y_pred)
    perf_dict['test']['Overall Performance']['precision'] = prec
    perf_dict['test']['Overall Performance']['recall'] = recall
    perf_dict['test']['Overall Performance']['f1'] = f1 #f1_score(Y_test_data, Y_pred, average=avg_mode)
    perf_dict['test']['Overall Performance']['report'] = classification_report(Y_test_data, Y_pred)
    output.output_json_readable(perf_dict['test']['Overall Performance'])

    plt.clf()
    cm = confusion_matrix(Y_test_data, Y_pred, normalize='all')
    sns.heatmap(cm, annot=True,annot_kws={"size":8})
    plt.tight_layout()
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    #plt.savefig(os.path.join(output_dir, 'perf_cm-{}-{}.png'.format(experiment_name, model_name)))

    #PS
    repredict=False
    if per_sample:
        if repredict:
            for idx in Y_test.index:
                print(X_test.loc[[idx]].values)
                print(X_test.loc[[idx]])
                single_pred_nd = model.predict(X_test.loc[[idx]].values)
                print(single_pred_nd)
                single_pred = single_pred_nd[0]
                correct = int(Y_test.loc[idx] == single_pred)
                perf_dict['test']['Per-Sample Performance'][idx]={'accuracy': correct}
        else:
            for i, idx in enumerate(Y_test.index):
                correct = int(Y_test.iloc[i] == Y_pred[i])
                perf_dict['test']['Per-Sample Performance'][idx]={'accuracy': correct}

    
    data.output_json_readable(perf_dict, write_json=True, file_path=os.path.join(output_dir, filename))

    return perf_dict, filename

def model_suite(X_train:pd.DataFrame, Y_train:pd.Series, X_test:pd.DataFrame, Y_test:pd.Series, experiment_name, scaler:StandardScaler=None, per_sample=True, perform_shap=False, output_dir='', notrain=False):
    X_train_data, Y_train_data, scaler_after = data.format_df_nd_all(X_train, Y_train, scaler)

    class_weight={0: 0.581, 1: 3.57}

    if notrain:
        print("SKIPPING TRAINING:")
        model_lr=None; model_gnb=None;model_rf=None;model_knn=None;model_svm=None
        psp_lr, perf_filename_lr = ovr_ps_perf(model_lr, X_test, Y_test, experiment_name, model_name='lr', output_dir=output_dir, scaler=scaler, notrain=notrain)
        psp_gnb, perf_filename_gnb = ovr_ps_perf(model_gnb, X_test, Y_test, experiment_name, model_name='gnb', output_dir=output_dir, scaler=scaler, notrain=notrain)
        psp_rf, perf_filename_rf = ovr_ps_perf(model_rf, X_test, Y_test, experiment_name, 'rf', output_dir=output_dir, scaler=scaler, notrain=notrain)
        psp_rf, perf_filename_kn = ovr_ps_perf(model_knn, X_test, Y_test, experiment_name, 'knn', output_dir=output_dir, scaler=scaler, notrain=notrain)
        psp_svm, perf_filename_svm = ovr_ps_perf(model_svm, X_test, Y_test, experiment_name, 'svm', output_dir=output_dir, scaler=scaler, notrain=notrain)
        
        return perf_filename_lr, perf_filename_gnb, perf_filename_rf, perf_filename_kn, perf_filename_svm
    
    #LR = LogisticRegression(solver='sag', C=1, class_weight='balanced', penalty='l2')
    LR = LogisticRegression()
    model_lr = LR.fit(X_train_data, Y_train_data)
    psp_lr, perf_filename_lr = ovr_ps_perf(model_lr, X_test, Y_test, experiment_name, model_name='lr', output_dir=output_dir, scaler=scaler, notrain=notrain)

    #GNB = GaussianNB(var_smoothing=0.0000433)
    GNB = GaussianNB()
    model_gnb = GNB.fit(X_train_data, Y_train_data) 
    psp_gnb, perf_filename_gnb = ovr_ps_perf(model_gnb, X_test, Y_test, experiment_name, model_name='gnb', output_dir=output_dir, scaler=scaler, notrain=notrain)

    RF = RandomForestClassifier()
    #RF = RandomForestClassifier(criterion='log_loss', max_depth=10, max_features='sqrt', n_estimators=100, class_weight='balanced')#
    model_rf = RF.fit(X_train_data, Y_train_data)
    psp_rf, perf_filename_rf = ovr_ps_perf(model_rf, X_test, Y_test, experiment_name, 'rf', output_dir=output_dir, scaler=scaler, notrain=notrain)

    
    #KNN = KNeighborsClassifier(algorithm='auto', n_neighbors=5, leaf_size=50, weights='uniform')
    KNN = KNeighborsClassifier()
    model_knn = KNN.fit(X_train_data, Y_train_data)
    psp_rf, perf_filename_kn = ovr_ps_perf(model_knn, X_test, Y_test, experiment_name, 'knn', output_dir=output_dir, scaler=scaler, notrain=notrain)

    
    #SVM = SVC(kernel='poly', C=10, decision_function_shape='ovo', class_weight='balanced')
    SVM = SVC()
    model_svm = SVM.fit(X_train_data, Y_train_data)
    psp_svm, perf_filename_svm = ovr_ps_perf(model_svm, X_test, Y_test, experiment_name, 'svm', output_dir=output_dir, scaler=scaler, notrain=notrain)

    # SHAP
    if perform_shap:
        shap_calc(model_rf, X_test, Y_test)

    del SVM, KNN, RF, GNB, LR
    return perf_filename_lr, perf_filename_gnb, perf_filename_rf, perf_filename_kn, perf_filename_svm

def model_suite_parameters(X_train:pd.DataFrame, Y_train:pd.Series, X_test:pd.DataFrame, Y_test:pd.Series, tag, experiment_name, scaler:StandardScaler=None, per_sample=True, perform_shap=False, output_dir='', chosen_model=None):
    results = {'lr':None, 'knn':None, 'gnb': None, 'rf': None, 'svm': None}
    
    metric = 'recall'
    refit = True#False

    X_train_data, Y_train_data, scaler_after = data.format_df_nd_all(X_train, Y_train, scaler)
    X_test_data, Y_test_data, scaler_after = data.format_df_nd_all(X_test, Y_test, scaler)

    LR = LogisticRegression()
    params_lr = {'penalty': ['l1', 'l2', 'elasticnet'],
                 'C': [1, 10, 25, 50, 100, 1000],
                 'solver': ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                 'class_weight': ['balanced']}
    gs_lr = GridSearchCV(estimator=LR, param_grid=params_lr, cv=5, scoring=metric, return_train_score=True)
    gs_lr.fit(X_train_data, Y_train_data)
    
    pd.DataFrame(gs_lr.cv_results_).to_csv('param_lr.csv')
    print(gs_lr.best_params_)
    results['lr'] = gs_lr.best_params_
    output.output_json_readable(gs_lr.best_params_, write_json=True, file_path='gs_lr_{}.json'.format(tag))
    
    
    GNB = GaussianNB()
    params_gnb = {'var_smoothing': np.logspace(0,-9, num=100)}
    gs_gnb = GridSearchCV(estimator=GNB, param_grid=params_gnb, cv=5, scoring='accuracy', return_train_score=True, refit=refit)
    gs_gnb.fit(X_train_data, Y_train_data)
    print(gs_gnb.best_params_)
    results['gnb'] = gs_gnb.best_params_
    output.output_json_readable(gs_gnb.best_params_, write_json=True, file_path='gs_gnb_{}.json'.format(tag))
    pd.DataFrame(gs_gnb.cv_results_).to_csv('param_gnb.csv')

    #y_pred = gs_gnb.best_estimator_.predict(X_test_data)
    if not refit:
        gnb_opt = GaussianNB().set_params(gs_gnb.best_params)
        gnb_opt.fit(X_train_data, Y_train_data)
        y_pred = gnb_opt.predict(X_test_data)

    RF = RandomForestClassifier()
    params_rf = {'n_estimators': [100, 200, 500, 1000, 2000],
                 'criterion': ['entropy', 'log_loss'],
                 'max_depth': [None, 5, 10, 25, 30, 50],
                 'max_features': ['sqrt', 'log2'],
                 'class_weight': ['balanced']}
    gs_rf = GridSearchCV(estimator=RF, param_grid=params_rf, cv=5, scoring=metric, return_train_score=True)
    gs_rf.fit(X_train_data, Y_train_data)
    
    pd.DataFrame(gs_rf.cv_results_).to_csv('param_rf_{}.csv'.format(metric))
    print(gs_rf.best_params_)
    results['rf'] = gs_rf.best_params_
    output.output_json_readable(gs_rf.best_params_, write_json=True, file_path='gs_rf_{}.json'.format(tag))
    
    gs_rf_est = gs_rf.best_estimator_
    y_pred = gs_rf_est.predict(X_test_data)
    y_pred_prob = gs_rf_est.predict_proba(X_test_data)
    
    y_pred = gs_lr.best_estimator_.predict(X_test_data)

    print('acc', accuracy_score(Y_test_data, y_pred))
    print('prec', precision_score(Y_test_data, y_pred))
    print('rec', precision_score(Y_test_data, y_pred))

    #print('rocauc', roc_auc_score(Y_test_data, y_pred_prob[:,1]))

    print('GS RESULTS', classification_report(Y_test_data, y_pred))

    KNN = KNeighborsClassifier()
    params_knn = {'n_neighbors': [2,3,4], 
                  'weights': ['uniform', 'distance'], 
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute_force'],
                  'leaf_size': [20, 25, 30, 35, 50]}
    gs_knn = GridSearchCV(estimator=KNN, param_grid=params_knn, cv=2, scoring='precision')
    gs_knn.fit(X_train, Y_train)
    print(gs_knn.best_params_)
    results['knn'] = gs_knn.best_params_
    output.output_json_readable(gs_knn.best_params_, write_json=True, file_path='gs_knn_{}.json'.format(tag))

    SVM = SVC()
    params_svm = {'kernel': ['linear', 'rbf', 'poly'],
                  'C': [0.1,1,10,100],
                  'decision_function_shape': ['ovo', 'ovr']}
    gs_svm = GridSearchCV(estimator=SVM, param_grid=params_svm, cv=2, scoring='precision')
    gs_svm.fit(X_train, Y_train)
    print(gs_svm.best_params_)
    output.output_json_readable(gs_svm.best_params_, write_json=True, file_path='gs_svm_{}.json'.format(tag))
    results['svm'] = gs_svm.best_params_

    return results

def main():
    data_dir_og = '~/PROJECTS/dote_1070-1083/.datasets/uci/cdc_diabetes/'
    entire_df_cont = pd.read_csv(os.path.join(data_dir_og, 'diabetes_binary_health_indicators_BRFSS2015-idxd-dt_int.csv'))
    entire_df_cont = pd.read_csv(os.path.join(data_dir_og, 'diabetes_binary_5050split_health_indicators_BRFSS2015-idxd-dt_int.csv'))

    data_dir_bt_STATIC_1211 = '~/PROJECTS/dote_1070-1083/codex-use_cases/cdc_diabetes/_runs/1210-propfreqcov-balanced_test'
    data_dir_bt_STATIC_1211 = '~/PROJECTS/dote_1070-1083/codex-use_cases/cdc_diabetes/_runs/1220-propfreqcov-balanced_test_label_included'
    data_dir_bt_STATIC_1211 = '~/PROJECTS/dote_1070-1083/codex-use_cases/cdc_diabetes/_runs/0121-propfreqcov-balanced_test_label_included_even'

    trainpool_df_STATIC_1211 = pd.read_csv(os.path.join(data_dir_bt_STATIC_1211, 'trainpool.csv'), index_col='ID')
    test_df_STATIC_1211 = pd.read_csv(os.path.join(data_dir_bt_STATIC_1211, 'test.csv'), index_col='ID')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    #250k
    train_df = entire_df_cont.loc[entire_df_cont.index.isin(trainpool_df_STATIC_1211.index.tolist())]

    #1500
    test_df = entire_df_cont.loc[entire_df_cont.index.isin(test_df_STATIC_1211.index.tolist())]
    drop_list=[]

    random = False #(input("RANDOM SPLIT (Y): ") == 'Y')
    if not random:
        X_train, X_test, Y_train, Y_test, split_filename = data.prep_split_data(None, train_df=train_df, test_df=test_df, name='params', drop_list=drop_list,
                                                                                    split_dir='./', 
                                                                                    target_col='Diabetes_binary', id_col='ID')
        assert X_train.index.tolist() == Y_train.index.tolist()
        assert X_test.index.tolist() == Y_test.index.tolist()

        print(X_train.shape)
        print(X_test.shape)
        print(Y_train.value_counts())
        print(Y_test.value_counts())
    else:
        entire_cont_sampled = entire_df_cont.sample(25000).set_index('ID')
        entire_cont_sampled_proc = entire_cont_sampled.drop(['Diabetes_binary'], axis=1)
        X_train, X_test, Y_train, Y_test = train_test_split(entire_cont_sampled_proc.values, entire_cont_sampled_proc['Diabetes_binary'].values, test_size=0.2)


    entire_df_data = entire_df_cont.drop(['Diabetes_binary', 'ID'],axis=1)
    print("DATA SHAPE", entire_df_data.values.shape)
    print(entire_df_cont['Diabetes_binary'].value_counts())
    scaler = StandardScaler().fit(entire_df_data.values)
    scaler = None
    
    #results = model_suite_parameters(X_train, Y_train, X_test, Y_test,experiment_name='params', scaler=scaler, tag='25k-auc')
    model_suite(X_train, Y_train, X_test, Y_test, 'TEST_PERF_0121', scaler=scaler)

if __name__ == '__main__':
    main()