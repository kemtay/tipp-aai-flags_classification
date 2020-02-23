#to ignore all scikit-learn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import ml_plot as mplt
import urllib.request as urllib
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import compress
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Binarizer
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.model_selection import cross_validate, cross_val_score, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

def get_model_name(model):
    mod_name = re.search('(.+?)\(', str(model)).group(1)  #extract the model name

    return mod_name

def decode_religion(keys):
    dict_religions = {0:'Catholic', 
                      1:'Other Christian', 
                      2:'Muslim', 
                      3:'Buddhist', 
                      4:'Hindu', 
                      5:'Ethnic', 
                      6:'Marxist', 
                      7:'Others'}
    list_val = []
    for k in keys:
        list_val.append(dict_religions[k])
        
    return list_val

class Flags:
    """ To classify a target based on country Flags """
    
    le = LabelEncoder()
    
    def __init__(self, target):  
        self.target = target
        return None
        
    def read_file(self, data_file, cols):    
        """ To read the data file to pandas dataframe """

        print("Reading data from data file %s..." %(data_file))
    
        self.df_flags = pd.read_csv(data_file, names=cols)
        print(self.df_flags.columns)
     
        print("File reading to dataframe completed.")
    
    def perform_eda(self, list_labels, list_binary_features, list_categorical_features, list_discrete_features):
        """ To perform Exploratory Data Analysis """
    
        pred_target = self.target 
        df = self.df_flags
    
        #explore the frequency on the target
        mplt.plot_hist(df[pred_target], pred_target, list_labels)
        
        #explore the discrete features for each religion
        for f in list_discrete_features:
            mplt.plot_bar_singlexy(df, f, pred_target, list_labels)
        
        #explore all binary features for each religion
        mplt.plot_hbar_by_sum(df, list_binary_features, list_labels, pred_target)
    
        #explore the categorical features for each religion
        for f in list_categorical_features:
            mplt.plot_bar_groupby_pivot(df, f, pred_target)    


    def drop_class(self, df, col_name, col_value):
        """ To drop a class by column name and value """
        
        print("Dropping %s from %s..." %(col_value, col_name))
        #print(df.shape)
        df.drop(df[df[col_name] == col_value].index, inplace = True)
        #print(df.shape)
    
    def drop_outliers(self, df, target_name, target_value, feature_name, feature_value):
        """ To drop outliers by target and feature """
        
        print("Dropping %s from %s where %s=%s..." %(feature_value, feature_name, target_name, target_value))
        outliers = df[(df[target_name] == target_value) & (df[feature_name] == feature_value)].index
        print(outliers)
        df.drop(outliers, inplace=True)
        
    def encode_data(self, df_f):
        """To prepare the data for predictive modeling"""
        
        print("Applying LabelEncoding to feature with categorical data...")
        df_cat = df_f.select_dtypes(include=['object'])
        for f in df_cat:
            print("Label-encoding %s..." %f)
            #print(df_dis[f].value_counts())
            df1 = pd.DataFrame(df_cat[f])  #create a new df for the feature
            le = LabelEncoder()
            np_le = le.fit_transform(df1[f])  #LabelEncoder returns np array
            f_col = f + "_le"
            df_le = pd.DataFrame(data=np_le.ravel(), columns=[f_col])  #flatten np array to create a new df
            print("Encoded labels: ", le.classes_)
            df_f.drop([f], axis=1, inplace=True)  #drop the original columns
            df_f = pd.concat([df_f,df_le], axis=1)
            #print(df_f.info())            
        
        print("Applying Binarizer to features with discrete data to 0 and 1...")
        df_dis = df_f.select_dtypes(include=['int64'])
        for f in df_dis:
            if df_dis[f].max() > 1:
                print("Binarizing %s ..." %f)
                df1 = pd.DataFrame(df_dis[f])  #create a new df for the feature
                #print(df1)
                f_col = f + "_bin"
                if f == 'colours':
                    np_bin = Binarizer(4).fit_transform(df1[f].values.reshape(1,-1))  #binarized with 1-4=0 and >4=1    
                else:
                    np_bin = Binarizer(0).fit_transform(df1[f].values.reshape(1,-1))
                #np_bin = bin_results.ravel()  #flatten the numpy array
                df_bin = pd.DataFrame(data=np_bin.ravel(),columns=[f_col])
                #print(df_bin)
                #df_f.drop([f], axis=1, inplace=True)  #drop the original columns
                df_f = pd.concat([df_f,df_bin], axis=1)     
    
        return df_f

    def prep_data(self, list_features):
        """ To prepare the data for training models """
        
        print("Preparing data for training models...")
        
        df_t = self.df_flags[self.target]
        df_f = self.df_flags[list_features]
        df_features = self.encode_data(df_f)
        #print("target:", df_t.info)
        #print("\nfeatures:", df_f.info)
        df_encoded = pd.concat([df_t, df_features], axis=1)
        print(df_encoded.info())
        
        self.drop_class(df_encoded, 'religion', 7)
        #self.drop_outliers(df_encoded, 'religion', 0, 'crosses', 1)
        #self.drop_outliers(df_encoded, 'religion', 3, 'crosses', 1)
        #self.drop_outliers(df_encoded, 'religion', 6, 'crescent', 1)
        
        self.df_flags_cleaned = df_encoded
        print(self.df_flags_cleaned.shape)
        print(self.df_flags_cleaned.info())
        #self.df_flags_cleaned.to_csv('flags_data_cleaned.csv')
        
        print("Data preparation completed.")
        
    def train_model(self, features, model):
        mod_name = get_model_name(model)
        print("Training model with %s ...." %mod_name)
        
        df = self.df_flags_cleaned
        target = self.target
        
        #split train-test dataset
        #print("Features:", features)
        X_data = df[features]
        y_data = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
        #print(len(X_train), len(X_test))
        
        #train the data using the selected model
        tr_model = model.fit(X_train, y_train)
        y_pred = tr_model.predict(X_test)
        
        cfs_mat_labels = np.unique(y_test)
        #print("confusion_matrix_labels:", cfs_mat_labels)
        test_acc = accuracy_score(y_test, y_pred)
        #test_auc = roc_auc_score(y_test, y_pred)
        cfs_mat = confusion_matrix(y_test, y_pred, labels=cfs_mat_labels)
        clf_rep = classification_report(y_test, y_pred)
        #transform confusion matrix to df for easier plotting
        df_cm = pd.DataFrame(cfs_mat, index=cfs_mat_labels, columns=cfs_mat_labels)
        #print("Test accuracy:", test_acc)
        #print("Confusion matrix:\n", df_cm)
        #print("Classification report:\n", clf_rep)
        
        cv_acc_scores = cross_validate(model, X_data, y_data, cv=30, scoring='accuracy', return_train_score='True')
        #cross_val_s = cross_val_score(model, X_data, y_data, cv=30)
        #print("Cross validation score:", cross_val_s, cross_val_s.mean())
        
        return tr_model, test_acc, df_cm, clf_rep, cv_acc_scores
    
    def select_features_rfe(self, clf_models, list_features):
        """ Perform RFE for feature selection """
        X = self.df_flags_cleaned[list_features]
        y = self.df_flags_cleaned[self.target]

        dict_test_acc = {get_model_name(model): [] for model in clf_models} #initialise dictionary for test scores per model
        dict_cv_acc = {get_model_name(model): [] for model in clf_models} #initialise dictionary for cv score per model
        list_mod_name = []
        min_f = 7
        max_f = 23
        best_acc_score = 0
        for model in clf_models:
            mod_name = get_model_name(model)
            print("Feature selection for model:", mod_name)
            if mod_name == 'KNeighborsClassifier':  #skip KNN as it does not support RFE
                continue
            list_mod_name.append(mod_name)
            max_acc_score = 0
            for i in range(min_f, max_f):
                list_test_acc, list_cv_acc = [], [] 
                rfe_selector = RFE(model, i, step=1)
                f_selected = rfe_selector.fit(X, y)
                #print(f_selected.support_)
                #print(f_selected.ranking_)
                list_selected_features = list(compress(list_features, f_selected.support_))  #return the f_selected features from list_features
                #print("List of the", i, "selected features:", list_selected_features)
                trained_model, test_acc, df_cm, clf_rep, cv_acc_scores = self.train_model(list_selected_features, model)
                dict_test_acc[mod_name].append(test_acc)
                dict_cv_acc[mod_name].append(cv_acc_scores['test_score'].mean())
                if test_acc > max_acc_score:
                    max_acc_score = test_acc
                    feature_num = i
                    selected_features = list_selected_features
            print("Maximum scores for %s: %.3f with" %(mod_name, max_acc_score))
            print(feature_num, "features selected:", selected_features)

        x_axis = np.arange(min_f, max_f)
        x_label = 'Number of features selected by RFE'
        y_label = 'Test accuracy score'
        g_title = 'Feature selection by RFE'
        mplt.plot_line_w_dict(dict_test_acc, list_mod_name, x_axis, x_label, y_label, g_title)

    def select_feature_corr(self, df, corr, corr_score):
        """ To select the features with low correlations with others """
        corr_len = len(corr)
        columns = np.full(corr_len, True, dtype=bool)  #initialize a numpy array with True (selected)
        for i in range(corr_len):  #loop the corr matrix by row
            for j in range(i+1, corr_len):  #loop the corr matrix by column and skip column-1 as it is itself
                if corr.iloc[i,j] >= corr_score:  #process the corr score of the pair which is greater than the input scores
                    #print("corr.iloc[i,j]", corr.iloc[i,j])
                    if columns[j]:  #check if the columns is true (selected), set it to False (unselect)
                        columns[j] = False
        selected_columns = df.columns[columns]
        #print("Features with low correlations with others:", selected_columns)
        
        return selected_columns.values
    
    def examine_corr(self, list_f):
        """ To examine correlations as part of feature selection """
        
        #correlation
        df_cleaned = self.get_df_cleaned()
        target = self.target
        corr_matrix = df_cleaned.corr().abs()
        #display top correlated features
        us_cm = corr_matrix.unstack()
        us_cm_not_self = us_cm[us_cm.values != 1]
        sorted_cm = us_cm_not_self.sort_values(kind="quicksort", ascending=False)
        np_high_corr_w_targets = sorted_cm[target][0:10].index.values
        print("Top 10 features correlate with target:\n ", np_high_corr_w_targets)
        #print("Top 20 correlated feature-pairs:\n", sorted_cm[sorted_cm > 0.4])  #display the uncorrelated feature-pairs based on corr scores
        #print("Top 20 correlated feature-pairs:\n", sorted_cm[12:80])  #display the top correlated feature-pairs by order
        
        #remove target from corr matrix to exclude it from feature selection
        corr_matrix_features = corr_matrix.drop([target], axis=1)  #drop target column
        corr_matrix_features.drop([target], axis=0, inplace=True)  #drop target row
        np_selected_features = self.select_feature_corr(df_cleaned[list_f], corr_matrix_features, 0.4)  #0.34 because it is half of the highest scores
        print("Features with low correlations with others:", np_selected_features) 
        
        #plot the correlation chart
        ylabel='Dependent variables'+self.target
        mplt.plot_corr(corr_matrix, ylabel)
        
        corr_selected_features = ['bars', 'stripes', 'colours_bin', 'red', 'green', 'blue', 'white', 'mainhue_le', 'black',
                                  'circles_bin', 'crosses_bin', 'sunstars_bin', 'crescent', 'triangle', 'topleft_le']
        
        print("Correlation selected features:", corr_selected_features)
        
        self.corr_selected_features = corr_selected_features
    
    def grid_search(self, model, param_grid, gscv_models):
        """ Perform GridSearchCV """
        
        X = self.df_flags_cleaned[self.corr_selected_features]
        y = self.df_flags_cleaned[self.target]
    
        mod_name = get_model_name(model)
        
        print("Performing GridSearchCV for model %s..." %mod_name)
        grid = GridSearchCV(model, param_grid, cv = 20, scoring = 'accuracy', return_train_score=True)
        grid.fit(X,y)
        #print("The best model: ", grid.best_estimator_)
        print("The best scores: %.3f" %grid.best_score_)
        print("The best cv results (train_score): %.3f" %(grid.cv_results_['mean_train_score'][grid.best_index_]))
        gscv_models.append(grid.best_estimator_)
        
        #return gscv_models

    def train_model_with_sss(self, clf_models):
        """ Train models with StratifiedShuffleSplit """
        
        print("Training models with StratifiedShuffleSplit...")
        
        df = self.df_flags_cleaned
        X = df[self.corr_selected_features].values
        y = df[self.target].values
        #print("Shape of X and y:", X.shape, y.shape)
        splits_count = 10
        sss = StratifiedShuffleSplit(n_splits=splits_count, test_size=0.3, random_state=42)
        #print(sss.get_n_splits(X, y))
        print(sss)
    
        dict_test_acc = {get_model_name(model): [] for model in clf_models} #initialise dictionary for test scores per model
        dict_cv_acc = {get_model_name(model): [] for model in clf_models} #initialise dictionary for cv score per model
        list_mod_name = [get_model_name(model) for model in clf_models]
        
        for train_index, test_index in sss.split(X, y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #print("counts:", len(X_train), len(y_train))
            #print("samples:", X_train[0:5], y_train[0:5])
            #train the data using the selected model
            for model in clf_models:
                mod_name = get_model_name(model)
                #print("StratifiedShuffleSplit for model:", mod_name)
                tr_model = model.fit(X_train, y_train)
                cv_scores = cross_val_score(model, X_train, y_train, cv=30)
                #print("Mean CV scores:", cv_scores.mean())
                dict_cv_acc[mod_name].append(cv_scores.mean())
                y_pred = tr_model.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)
                dict_test_acc[mod_name].append(test_acc)
                #print(dict_test_acc)
    
        #plot line chart for the test scores
        #print("\nThe best model is %s with the highest accuracy score: %.3f" %(best_mod, best_acc_score))
        x_axis = np.arange(1, splits_count+1)
        x_label = 'n_splits'
        y_label = 'Test accuracy score'
        g_title = 'Test scores with StratifiedShuffleSplit()'
        mplt.plot_line_w_dict(dict_test_acc, list_mod_name, x_axis, x_label, y_label, g_title)
        
    
    def get_df(self):
        return self.df_flags
    
    def get_df_cleaned(self):
        return self.df_flags_cleaned
    
    def get_corr_selected_features(self):
        return self.corr_selected_features
    
