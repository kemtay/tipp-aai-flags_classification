"""
Name:   TAY SOK KEM
Course: TIPP AAI Intake 1 by Republic Polytechnic
Module: Machine Learning Fundamentals
Submission deadline: 19 Feb 2020 23:59
 
"""

import FlagsClassifier as fclf
import ml_plot as mplt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def main():
    cols = ['name', 'continent', 'zone', 'area', 'population', 'language', 'religion', 
            'bars', 'stripes', 'colours', 'red', 'green', 'blue', 'gold', 'white', 'black', 
            'orange', 'mainhue', 'circles', 'crosses', 'saltires', 'quarters', 'sunstars', 
            'crescent', 'triangle', 'icon', 'animate', 'text', 'topleft', 'botright']

    list_labels = ['Catholic', 'Other Christian', 'Muslim', 'Buddhist', 'Hindu', 'Ethnic', 'Marxist', 'Others']
    list_all_features = ['bars', 'stripes', 'colours', 'red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 
                         'mainhue', 'circles', 'crosses', 'saltires', 'quarters', 'sunstars', 'crescent', 
                         'triangle', 'icon', 'animate', 'text', 'topleft', 'botright']
    list_all_encoded_features = ['bars_bin', 'stripes_bin', 'colours_bin', 'red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 
                                 'mainhue_le', 'circles_bin', 'crosses_bin', 'saltires', 'quarters_bin', 'sunstars_bin', 
                                 'crescent', 'triangle', 'icon', 'animate', 'text', 'topleft_le', 'botright_le']

    list_binary_features = ['red', 'green', 'blue', 'gold', 'white', 'black', 'orange', \
                            'crescent', 'triangle', 'icon', 'animate', 'text']
    list_categorical_features = ['mainhue', 'topleft', 'botright']
    list_discrete_features = ['bars', 'stripes', 'colours', 'circles', 'crosses', 'saltires', 'quarters', 'sunstars']
    
    #to initialize FlagsClassifier with chosen target    
    flags = fclf.Flags(target='religion')
    
    #to read file to pandas dataframe
    data_file = 'flag_data.csv'
    flags.read_file(data_file, cols)
    print(flags.get_df().info())

    #to perform Data Exploratory Analysis (EDA)
    flags.perform_eda(list_labels, list_binary_features, list_categorical_features, list_discrete_features)
    
    #to prepare the data for training models
    flags.prep_data(list_all_features)
    df_cleaned = flags.get_df_cleaned()
    
    #to train models with all encoded features
    my_clf_models = [KNeighborsClassifier(n_neighbors=15, weights='distance'),
                     DecisionTreeClassifier(max_depth=15, random_state=42),
                     RandomForestClassifier(n_estimators=20, max_depth=15, min_samples_split=2, random_state=42),
                     AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=5, random_state=42),
                     BernoulliNB(), MultinomialNB(),
                     LinearSVC(max_iter=50, random_state=42)]
    for model in my_clf_models:
        trained_model, test_acc, df_cm, clf_rep, cv_acc_scores = flags.train_model(list_all_encoded_features, model)
        mod_name = fclf.get_model_name(model)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        plt.rcParams.update({'font.size': 12})
        mplt.plot_cfm(df_cm, mod_name, test_acc, ax1)
        mplt.plot_cv(cv_acc_scores, ax2)
        plt.show()
    
    #to select features using Recursive Feature Elimination (RFE)
    flags.select_features_rfe(my_clf_models, list_all_encoded_features)
    
    #examine correlation for all features columns
    df_f =df_cleaned.drop(flags.target, axis=1)
    all_feature_cols = list(df_f.columns)
    #print("all_feature_cols:", all_feature_cols)
    flags.examine_corr(all_feature_cols)
    
    #to define param_grid for GridSearchCV 
    k_range = list(range(5,20))
    weight_options = ['uniform', 'distance']
    n_est = list(range(1,10))
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    my_gscv_param_grids = [dict(n_neighbors = k_range, weights = weight_options),  #KNN
                            dict(min_samples_split=range(10,15,20), max_depth=[10,15,20,25]),  #DTClf
                            dict(n_estimators=[10,15,20,25], max_depth=[5, 6, 7, 8, 9, 10]),  #RFClf
                            dict(base_estimator__criterion=['gini', 'entropy'], base_estimator__splitter=['best', 'random'], n_estimators=n_est), #abclf
                            dict(alpha=alphas, fit_prior=[True, False], class_prior=[None, [1,.9,.8,.6,.5,.7,.4]]),  #BNB
                            dict(alpha=alphas, fit_prior=[True, False], class_prior=[None, [1,.9,.8,.6,.5,.7,.4]]),  #MNB
                            dict(dual=[True,False], C=np.arange(0.01,100,10), class_weight=[None,'balanced'], max_iter=[10,25,50,75,100])  #LSVC
                            ]
    
    #to perform GridSearchCV for my_clf_model with corr_selected_features
    gscv_models = []   
    for i in range(len(my_clf_models)):
        flags.grid_search(my_clf_models[i], my_gscv_param_grids[i], gscv_models)
        
    print(gscv_models)
    best_acc = 0
    #to train GridSearchCV 'best_estimator_' with corr_selected_features
    corr_selected_features = flags.get_corr_selected_features()
    for model in gscv_models:
        trained_model, test_acc, df_cm, clf_rep, cv_acc_scores = flags.train_model(corr_selected_features, model)
        mod_name = fclf.get_model_name(model)
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = trained_model
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        plt.rcParams.update({'font.size': 12})
        mplt.plot_cfm(df_cm, mod_name, test_acc, ax1)
        mplt.plot_cv(cv_acc_scores, ax2)
        plt.show()
    print("The best model is:", best_model, "\nwith test accuracy score:", best_acc)
    
    #to train GridSearchCV 'best_estimator_' with gscv_models
    flags.train_model_with_sss(gscv_models)

# =============================================================================

main()
