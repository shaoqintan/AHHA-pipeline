import numpy as np
import pandas as pd
import os
import scipy
import time
import statsmodels.api as sm
import sys
import logging
import ast
import time
import gc
import pickle
import traceback
import pingouin as pg

from utils import *
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from statsmodels.base._penalties import L2
import time

def plot_reliability(df_odd, df_even, feature, filename):   
    
    df_odd = df_odd.loc[df_odd['ID'] > 100]
    df_even = df_even.loc[df_even['ID'] > 100]

    df_odd = df_odd.merge(df_even[['ID', 'Week']], on=['ID', 'Week'], how='inner')
    df_even = df_even.merge(df_odd[['ID', 'Week']], on=['ID', 'Week'], how='inner')

    df_even = df_even.drop_duplicates()
    df_odd = df_odd.drop_duplicates()

    
    M1 = pd.DataFrame(columns=['ratings', 'targets', 'rater'])
    M2 = pd.DataFrame(columns=['ratings', 'targets', 'rater'])


    M1['targets'] = df_odd['ID']
    M2['targets'] = df_even['ID']


    M1['ratings'] = df_odd[feature]
    M2['ratings'] = df_even[feature]

    M1['rater'] = 'Odd'
    M2['rater'] = 'Even'

    M = pd.concat([M1, M2])

    icc = pg.intraclass_corr(data=M, targets='targets', raters='rater',
                             ratings='ratings', nan_policy='omit').round(10)

    return icc.loc[icc['Type']== 'ICC3', 'ICC'].values[0]


def compute_feature_performance(df, df_odd, df_even, outer_train_id, test_idx):
    
    df = df.loc[df['ID'].isin(outer_train_id)]
    df_odd = df_odd.loc[df_odd['ID'].isin(outer_train_id)]
    df_even = df_even.loc[df_even['ID'].isin(outer_train_id)]
    
    df_label = pd.read_csv('Data/Acute_labels.csv')
    
    df_corr = df.drop(columns=['Week']).corr('spearman')[['ARAT', 'FM_score']]
    df_delta_corr = df_label.corr('spearman')[['ARAT', 'FM_score']]
    
    features = []
    all_icc = []
    all_corr_arat = []
    all_corr_fma = []
    all_delta_corr_arat = []
    all_delta_corr_fma = []
    all_auc_arat = []
    all_auc_fma = []
    
    for f in df.columns:
        if (f in ['ID', 'Week', 'ARAT', 'FM_score']):
            continue
            
        r = df_corr.loc[[f]].values[0]
        
        icc = plot_reliability(df_odd, df_even, f, '')
            
        delta_corr = df_delta_corr.loc[[f]].values[0]
        
        
        auc_arat = roc_auc_score(df_label['ARAT_label'], df_label[f])
        auc_fma = roc_auc_score(df_label['FM_score_label'], df_label[f])
        
                    
        features.append(f)
        all_icc.append(icc)
        
        all_corr_arat.append(r[0])
        all_corr_fma.append(r[1])
        
        all_delta_corr_arat.append(delta_corr[0])
        all_delta_corr_fma.append(delta_corr[1])
        
        all_auc_arat.append(auc_arat)
        all_auc_fma.append(auc_fma)
        
    df_results = pd.DataFrame({'Features':features, 'ARAT corr':all_corr_arat, 'FMA corr':all_corr_fma, 
                                              'ARAT resp':all_auc_arat, 'FMA resp':all_auc_fma, 'ARAT delta corr':all_delta_corr_arat, 'FMA delta corr':all_delta_corr_fma, 'ICC':all_icc})
    
    df_results['Mean'] = np.nanmean(np.abs(df_results[['ARAT corr', 'FMA corr', 'ARAT resp', 'FMA resp', 'ARAT delta corr', 'FMA delta corr', 'ICC']]), axis=1)
    cv_corr = np.std([df_results['ARAT corr'].values, df_results['FMA corr'].values], axis=0)/np.mean([df_results['ARAT corr'].values, df_results['FMA corr'].values], axis=0)
    cv_resp = np.std([df_results['ARAT resp'].values, df_results['FMA resp'].values], axis=0)/np.mean([df_results['ARAT resp'].values, df_results['FMA resp'].values], axis=0)
    cv_delta_corr = np.std([df_results['ARAT delta corr'].values, df_results['FMA delta corr'].values], axis=0)/np.mean([df_results['ARAT delta corr'].values, df_results['FMA delta corr'].values], axis=0)

    df_results['Mean_CV'] = np.mean([cv_corr, cv_resp, cv_delta_corr], axis=0)

    df_results['Overall'] = df_results['Mean']/df_results['Mean_CV']
    
    df_results.to_csv('Data/fs/Acute_features_all_results_{}.csv'.format(test_idx))
    
    
def filter_features(df, outer_train_id, test_idx):
    df = df.loc[df['ID'].isin(outer_train_id)]
    
    df_results = pd.read_csv('Data/fs/Acute_features_all_results_{}.csv'.format(test_idx))
    
    df_results = df_results.loc[(np.round(df_results['ARAT resp'], 2) > 0.7)&(np.round(df_results['FMA resp'], 2) > 0.7)]
    df_results = df_results.loc[(np.round(df_results['ARAT delta corr'], 2) > 0.4)&(np.round(df_results['FMA delta corr'], 2) > 0.4)]
    df_results = df_results.loc[(np.abs(np.round(df_results['ARAT corr'], 2)) > 0.75)&(np.abs(np.round(df_results['FMA corr'], 2) > 0.75))]
    df_results = df_results.loc[np.round(df_results['ICC'], 2) >= 0.75]
    df_results = df_results.sort_values(['Overall'], ascending=False)
    
    corr_matrix = df[df_results['Features']].corr().abs()
    
    threshold = 0.9

    # Create a boolean matrix where correlations above the threshold are True
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    high_corr_pairs = (corr_matrix > threshold) & upper_triangle

    # List to hold the names of features to drop
    to_drop = []

    # Iterate through the correlation matrix and find features to drop
    for col in high_corr_pairs.columns:
        if any(high_corr_pairs[col]):
            to_drop.append(col)

    df_reduced = df.drop(columns=to_drop)
    
    df_results = df_results.loc[df_results['Features'].isin(df_reduced.columns)]
    df_results.to_csv('Data/fs/Acute_features_top_results_{}.csv'.format(test_idx))
    
    df_results = df_results.rename(columns={'Features':'Method'})
    
    return df_results.sort_values(['Overall'], ascending=False)['Method'].values
    

def main():
           
    try:
    
        model_type = sys.argv[1]

        label = sys.argv[2]
                                        
        test_idx = ast.literal_eval(sys.argv[3])
        
            
        df_all_feature = pd.read_csv("Data/Acute Stroke Cohort features.csv", index_col=0)
        df_all_feature_odd = pd.read_csv("Data/Acute Stroke Cohort features_odd.csv")
        df_all_feature_even = pd.read_csv("Data/Acute Stroke Cohort features_even.csv")
        
        df_all_feature.dropna(subset=['FM_score', 'ARAT'], inplace=True)
        

        all_idx = df_all_feature['ID'].unique()


        outer_train_id = all_idx[all_idx != test_idx]

        inner_kfold = KFold(n_splits=len(outer_train_id), shuffle=True, random_state=seed)


        file = 'Acute_p2p_noCFS/Acute_p2p_{}/'.format(label)


        file += '{}/'.format(model_type)


        create_dir(file)


        file += 'cfs/'

        create_dir(file)

        file += '{}'.format(test_idx)
 
        
        # compute_feature_performance(df_all_feature, df_all_feature_odd, df_all_feature_even, outer_train_id, test_idx) # only need to run it one time
        
        df_features = filter_features(df_all_feature, outer_train_id, test_idx)

        
        feature_list = np.array([len(df_features)])
            

        if os.path.exists(file+'_model_noFE.pkl'):
            return

        if os.path.exists('{}.log'.format(file)):
            return

       
        alpha_list = [0.001, 0.01, 0.1]
            

        df_inner = pd.DataFrame(columns=['Inner_fold', 'Alpha', 'N', 'Features', 'val_loss'])
            

        for alpha in alpha_list:

            for inner_fold, (inner_train_id, val_id) in enumerate(inner_kfold.split(outer_train_id)):

                X_train, y_train, X_val, y_val, formula = get_processed_data(df_all_feature.copy(), df_features, outer_train_id[inner_train_id], outer_train_id[val_id], label)


                X_train['Sum'] = y_train

                model = sm.MixedLM.from_formula('Sum ~ {}'.format(formula), X_train, groups=X_train['ID'])

                l2 = L2(weights=alpha)
                model = model.fit(method=["powell", "lbfgs"], fe_pen=l2)

                y_pred = model.predict(X_val)

                val_loss = mean_squared_error(y_val, y_pred)


                del X_train, y_train, X_val, y_val, y_pred, model


                df_inner = pd.concat([df_inner, pd.DataFrame({'Inner_fold':inner_fold, 'Alpha':alpha, 'val_loss':val_loss}, index=[0])], ignore_index=True)


                df_inner.to_csv(file+'_inner.csv')

                gc.collect()

        
        df_inner = df_inner.drop(columns=['Inner_fold', 'Features']).groupby(['Alpha']).mean().drop_duplicates()
        df_inner.to_csv(file+'_inner.csv')
        

        df_inner = pd.read_csv(file+'_inner.csv')
        min_idx = df_inner['val_loss'].idxmin()
        alpha = df_inner.iloc[min_idx].values[0]


        X_train, y_train, X_test, y_test, formula = get_processed_data(df_all_feature, df_features, outer_train_id, [test_idx], label)


        X_train['Sum'] = y_train

        model = sm.MixedLM.from_formula('Sum ~ {}'.format(formula), X_train, groups=X_train['ID'])

        l2 = L2(weights=alpha)
        model = model.fit(method=["powell", "lbfgs"], fe_pen=l2)
        
        y_pred_train = model.predict(X_train)
        train_loss = mean_squared_error(y_train, y_pred_train)
        
        y_pred_test = model.predict(X_test)
        test_loss = mean_squared_error(y_test, y_pred_test)
        
        df_inner['train_loss'] = train_loss
        df_inner['test_loss'] = test_loss
        
        
        df_inner.to_csv(file+'_inner.csv')
        

        with open('{}_model_noFE.pkl'.format(file), 'wb') as f:
            pickle.dump(model, f)

            

    except Exception as e:
        with open('{}.log'.format(file), 'a') as file_open:
            
            file_open.write(f"An error occurred: {str(traceback.format_exc())}\n")
            file_open.write(f"Parameters are: alpha:{alpha}, top{top}\n")
    
if __name__ == "__main__":
    main()
