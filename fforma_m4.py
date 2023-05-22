#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from ESRNN.utils_evaluation import evaluate_prediction_owa

from fforma import FFORMA
##### if want to using R_Models
# from fforma.r_models import (
#     ARIMA,
#     ETS,
#     ThetaF,
#     Naive,
#     SeasonalNaive
# )
from fforma.meta_model import (
    MetaModels,
    temp_holdout,
    calc_errors,
    get_prediction_panel
)
from ESRNN.m4_data import prepare_m4_data, seas_dict
from tsfeatures import tsfeatures
# from tbats import TBATS, BATS
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.croston import Croston
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
freqs = {'Quarterly': 4,
         'Yearly': 1}
# freqs = {'Hourly': 24, 'Daily': 1,
#          'Monthly': 12, 'Quarterly': 4,
#          'Weekly':1, 'Yearly': 1}
# Weekly
# ===============  Model evaluation  ==============
# OWA: 0.915
# SMAPE: 7.268
# MASE: 2.476
#
###
# ===============  Model evaluation  ==============
# OWA: 0.848
# SMAPE: 6.939
# MASE: 2.229
# Hourly
#
#
# ===============  Model evaluation  ==============
# OWA: 0.797
# SMAPE: 18.609
# MASE: 1.395
# 1111111111111111 Daily

def prepare_to_train_fforma(dataset, validation_periods, seasonality):

    X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset, './data', 100000)

    # Preparing errors
    y_holdout_train_df, y_val_df = temp_holdout(y_train_df, validation_periods)

    meta_models = {
        ##################使用Python的包
        'Naive': NaiveForecaster(strategy="last"),
        'Naive1': NaiveForecaster(strategy="drift"),
        'Naive3': NaiveForecaster(strategy="mean"),
        'Theta': ThetaForecaster(sp=seasonality),
        'ExponentialSmoothing':  ExponentialSmoothing(trend='add', seasonal='multiplicative', sp=seasonality),
        'ETS':  AutoETS(auto=False, sp=seasonality, n_jobs=1),
        # 'AutoETS':  AutoETS(auto=True, sp=seasonality, n_jobs=1),
        'Croston': Croston(smoothing=0.7),
        ############### 使用 R语言的包 ,确保自己本地已经安装R语言和forecast包
        # 'RThetaF': ThetaF(freq=seasonality),
        # 'RNaive': Naive(freq=seasonality),
        # 'RSeasonalNaive': SeasonalNaive(freq=seasonality),
        # 'RNaive2': Naive2(seasonality=seasonality)
    }
    validation_meta_models = MetaModels(meta_models)
    validation_meta_models.fit(y_holdout_train_df)
    meta_models_prediction = validation_meta_models.predict(y_val_df) # 预测这段期间的数据

    #Calculating errors
    meta_models_prediction = pd.merge(meta_models_prediction,y_val_df,on=['unique_id','ds'])
    errors = calc_errors(meta_models_prediction, y_holdout_train_df, seasonality,benchmark_model='Naive')

    #Calculating features
    features = tsfeatures(y_holdout_train_df, seasonality)
    # features = tsfeatures(y_train_df, seasonality)

    #Calculating actual predictins
    meta_models = MetaModels(meta_models)
    meta_models.fit(y_train_df)

    predictions = meta_models.predict(y_test_df[['unique_id', 'ds']])

    return errors, features, predictions
def evaluate_fforma(dataset_name, fforma_df, directory, num_obs):
    print(dataset_name)
    _, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name=dataset_name,
                                                          directory=directory,
                                                          num_obs=num_obs)

    y_test_fforma = fforma_df[fforma_df['unique_id'].isin(y_test_df['unique_id'].unique())]
    y_test_fforma = y_test_fforma.rename(columns={'fforma_prediction': 'y_hat'})
    y_test_fforma = y_test_fforma.filter(items=['unique_id', 'ds', 'y_hat'])

    seasonality = freqs[dataset_name]
    owa, mase, smape = evaluate_prediction_owa(y_test_fforma, y_train_df, X_test_df, y_test_df, seasonality)

    return dataset_name, owa, mase, smape

def main():
    # for iter_datasets in freqs.keys():
    for iter_datasets in ['Weekly']:
        complete_errors, complete_features, complete_predictions = [], [], []
        print(1111111111111111,iter_datasets)
        for dataset in ['Weekly']: #'Daily', etc
            validation_periods = seas_dict[dataset]['output_size']
            seasonality = seas_dict[dataset]['seasonality']
            errors, features, predictions = prepare_to_train_fforma(dataset, validation_periods, seasonality)

            complete_errors.append(errors)
            complete_features.append(features)
            complete_predictions.append(predictions)

        complete_errors = pd.concat(complete_errors)
        complete_features = pd.concat(complete_features)
        complete_predictions = pd.concat(complete_predictions)
        print('Training fforma')

        #Training fforma
        optimal_params = {'n_estimators': 94,
                          'eta': 0.58,
                          'max_depth': 14,
                          'subsample': 0.92,
                          'colsample_bytree': 0.77}
        fforma = FFORMA(params=optimal_params)
        fforma.fit(errors=complete_errors,
                   holdout_feats=complete_features.set_index('unique_id'),
                   feats=complete_features.set_index('unique_id'))

        fforma_predictions = fforma.predict(complete_predictions.set_index('unique_id'))
        print(fforma_predictions)
        #evaluate predictions
        fforma_predictions.reset_index().to_csv('./data/fforma_predictions.csv',index=False)
        # evaluate_fforma(dataset_name, fforma_df, directory, num_obs):
        from functools import partial
        import multiprocessing as mp
        # evaluate_fforma_p = partial(evaluate_fforma, fforma_df=fforma_predictions.reset_index(), directory='./data', num_obs=100000)
        # for dataset_name in freqs.keys():Weekly
        for dataset_name in [iter_datasets]:
            print('evaluate dataset_name',dataset_name)
            evaluate_fforma(dataset_name, fforma_predictions.reset_index(), './data', 100000)


if __name__=='__main__':
    main()
