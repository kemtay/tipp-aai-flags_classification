"""
Machine Learning plots

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hist(df, x_col, x_labels):
    """ To plot histogram for frequency counts """
    
    print("Plotting histogram for %s..." %x_col)
    print("Total counts:", len(df))
    uniq_labels = sorted(df.unique())
    x_labels = x_labels
    plt.figure(figsize=(6, 6))
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 15})
    x_pos = np.arange(len(df))  #the label locations
    ax = sns.countplot(df)
    ax.set_xticklabels(x_labels, rotation=40, ha='right')
    plt.tight_layout()
    for v in range(len(uniq_labels)):
        plt.text(v, (df == v).sum(), str((df == v).sum()))
    plt.show()
    
def plot_hbar_by_sum(df, x_axis, y_axis, y_col):
    """ To plot bar chart for data aggregated by sum """
    
    print("Plotting horizontal bar chart aggregated by sum...")
    
    plt.rcParams.update({'font.size': 25})
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(40, 60))

    y_pos = np.arange(len(y_axis))  #the label locations
    y_labels = y_axis
    width = 0.08  #the width of the bars
    cluster = -(len(y_axis)/2)
    bar_colors = ['red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'magenta', 'grey', 'yellow', 'purple', 'cyan']
    for x in x_axis:
        idx = x_axis.index(x)
        x_data = df.groupby(y_col)[x].sum()
        ax.barh(y_pos-cluster*width, x_data, width, label=x, color=bar_colors[idx], edgecolor='navy', align='center')
        cluster += 1
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Counts')
    ax.set_title('Counts of all binary features of all %ss' %y_col)
    
    plt.legend(loc='upper right')
    plt.show()
    
def plot_bar_groupby_pivot(df, feature, target):
    """ To plot bar chart for grouped data with pivot """
    
    print("Plotting horizontal bar chart of %s group by %s ..." %(feature, target))
    
    uniq_feature = df[feature].unique()

    df1 = df.groupby([target,feature]).size().reset_index(name='count')
    #print(df1)
    df2 = df1.pivot(index=target, columns=feature, values='count')
    #print(df2)

    y_pos = np.arange(len(df2.index))
    width = 0.12
    cluster = -(len(df2.index)/2)
    plt.rcParams.update({'font.size': 15})
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12))
    for f in uniq_feature:
        ax.barh(y_pos-cluster*width, df2[f], width, label=f, color=f, edgecolor='navy', align='center')
        cluster += 1

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df2.index)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(feature)
    ax.set_title('%s by %s' %(feature, target))
    
    plt.legend(loc='upper right')
    plt.show()

def plot_bar_singlexy(df, feature, target, x_labels):
    """ To plot bar chart for single x against y """
    
    print("Plotting bar chart of %s by %s ..." %(feature, target))
    
    uniq_feature = df[feature].unique()

    df1 = df.groupby([target,feature]).size().reset_index(name='count')
    #print(df1)
    df2 = df1.pivot(index=target, columns=feature, values='count')
    #print(df2)
    
    uf_len = len(uniq_feature)
    index_len = len(df2.index)
    x_pos = np.arange(index_len)  #the label locations
    x_labels = x_labels
    #x_labels = decode_religion(df2.index.to_list())  #produce religion as labels on x-axis
    #print("Type of x_labels:", df2.index.to_list())
    
    
    fig_w = max(len(uniq_feature) * 2, 5)
    fig_h = max(len(uniq_feature) * 1, 3)
    font_s = max(len(uniq_feature) * 2, 10)
    width = min((fig_w/uf_len)+0.10, 0.15)  #the width of the bars
    cluster = -(index_len/2)
    
    plt.figure(figsize=(fig_w, fig_h))
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': font_s})

    for f in uniq_feature:
        plt.bar(x_pos-cluster*width, df2[f], width, label=f)
        cluster += 1
    
    plt.ylabel('Number of countries')
    plt.title('%s counts by %s' %(feature, target))
    plt.xlabel(target)
    plt.xticks(x_pos, x_labels, rotation=45)
    plt.legend(loc='upper right')
    plt.show()

def plot_cfm(df, mod_name, test_acc, ax=None):
    """ To plot confusion matrix """
    
    ax = ax
    ax.set_title("%s model \nAccuracy:%.3f" %(mod_name, test_acc))
    sns.heatmap(df, annot=True, fmt='d', ax=ax)
    ax.set_ylim(ax.get_ylim()[0]+0.5, 0)  #fix seaborn heatmaps problem in matplotlib 3.1.1
    ax.set_ylabel("Actual target")
    ax.set_xlabel("Predicted target")

def plot_cv(cv_acc_scores, ax):
    """ To plot cross validation """
    
    #print("Cross validation accuracy scores:", cv_acc_scores)   
    print("Cross validation test score average: %.3f" %(cv_acc_scores['test_score'].mean()))
    print("Cross validation train score average:%.3f" %(cv_acc_scores['train_score'].mean()))
    ax = ax
    y_pos = np.arange(len(cv_acc_scores['train_score']))
    #print("y_pos:", y_pos)
    ax.plot(y_pos, cv_acc_scores['train_score'], label='Training accuracy')
    ax.plot(y_pos, cv_acc_scores['test_score'], label='Validation accuracy')
    ax.set_xlabel('cv iterations')
    ax.set_ylabel('accuracy')
    plt.legend(loc='bottom right')

def plot_line_w_dict(dict_data, dict_keys, x_axis, x_label, y_label, g_title):
    """ To plot a line chart with dictionary data """
    #print(dict_data[dict_key])
    plt.figure(figsize=(20, 10))
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 22})
    plt.xlabel(x_label)
    plt.title(g_title)
    best_test_acc_score = 0
    best_mean_acc_score = 0
    for k in dict_keys:
        max_test_acc_score = max(dict_data[k])
        mean_test_acc_score = np.mean(dict_data[k])
        if max_test_acc_score > best_test_acc_score and mean_test_acc_score > best_mean_acc_score:
            best_test_acc_score = max_test_acc_score
            best_mean_acc_score = mean_test_acc_score
            best_model = k
        print("The highest test scores for model %s is %.3f, mean is %.3f" %(k, max_test_acc_score, mean_test_acc_score))
        plt.plot(x_axis.astype(str), dict_data[k], label=k)
    print("The best test accuracy score is: %.3f" %best_test_acc_score)
    print("The best test mean accuracy score is: %.3f" %best_mean_acc_score)
    print("The best model is:", best_model)
    plt.legend(loc='bottom left')
    plt.show()

def plot_corr(corr_matrix, ylabel):
    """ To plot a correlation chart """

    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))  #generate a mask for the upper triangle
    plt.figure(figsize=(20,20))
    plt.rcParams.update({'font.size': 15})
    ax = sns.heatmap(corr_matrix, mask=mask, square=True, annot=True)
    #print("ylim:", ax.get_ylim())
    ax.set_ylim(ax.get_ylim()[0]+0.5, 0)  #fix seaborn heatmaps problem in matplotlib 3.1.1
    plt.title('Correlation')
    plt.ylabel(ylabel)
    plt.show()

def plot_multiple_lines(dict_data, dict_keys, x_axis, x_label, y_label, g_title):
    #plot line chart for the test scores
    x_axis = np.arange(1, splits_count+1)
    plt.figure(figsize=(20, 10))
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 22})
    plt.xlabel('n_splits')
    plt.title('Test scores with StratifiedShuffleSplit()')
    for model in clf_models:
        mod_name = get_model_name(model)
        print("The highest CV scores for model %s is %.3f, mean is %.3f" %(mod_name, max(dict_cv_acc[mod_name]), np.mean(dict_cv_acc[mod_name])))
        print("The highest test scores for model %s is %.3f, mean is %.3f" %(mod_name, max(dict_test_acc[mod_name]), np.mean(dict_test_acc[mod_name])))
        plt.plot(x_axis.astype(str), dict_test_acc[mod_name], label=mod_name)
    plt.legend(loc='bottom left')
    plt.show()