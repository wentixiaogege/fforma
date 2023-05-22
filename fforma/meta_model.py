#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import dask
from collections import ChainMap
from functools import partial
from itertools import product
from copy import deepcopy

from sklearn.utils.validation import check_is_fitted
from ESRNN.utils_evaluation import smape, mase, evaluate_panel

class MetaModels:
    """
    Parameters
    ----------
    models: dict
        Dictionary of models to train. Ej {'ARIMA': ARIMA()}
    scheduler: str
        Dask scheduler. See https://docs.dask.org/en/latest/setup/single-machine.html
        for details.
        Using "threads" can cause severe conflicts.
    """
    def __init__(self, models, scheduler='processes'):
        self.models = models
        self.scheduler = scheduler
    ###https://zhuanlan.zhihu.com/p/478551556?utm_id=0
    def sktime_add_freq(self,train):
        perd = pd.infer_freq(train.index)
        using_p = perd
        if perd in ["MS", "M", "BM", "BMS"]:
            train.index = pd.PeriodIndex(train.index, freq="M")
            using_p = 'M'
        elif perd in ["BH", "H"]:
            train.index = pd.PeriodIndex(train.index, freq="H")
            using_p = 'H'
        elif perd == "B":
            train.index = pd.PeriodIndex(train.index, freq="B")
            using_p = 'B'
        elif perd == "D":
            train.index = pd.PeriodIndex(train.index, freq="D")
            using_p = 'D'
        elif perd in ["W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"]:
            train.index = pd.PeriodIndex(train.index, freq="W")
            using_p = 'W'
        elif perd in ["Q", "QS", "BQ", "BQS","Q-DEC","BQ-DEC","QS-OCT","BQS-OCT"]:
            train.index = pd.PeriodIndex(train.index, freq="Q")
            using_p = 'Q'
        elif perd in ["A", "BA", "AS", "BAS","A-DEC","BA-DEC","AS-JAN","BAS-JAN"]:
            train.index = pd.PeriodIndex(train.index, freq="A")
            using_p = 'A'
        elif perd in ["T", "min"]:
            train.index = pd.PeriodIndex(train.index, freq="m")
            using_p = 'm'
        elif perd == "S":
            train.index = pd.PeriodIndex(train.index, freq="s")
            using_p = 's'
        elif perd in ["L", "ms"]:
            train.index = pd.PeriodIndex(train.index, freq="L")
            using_p = 'L'
        elif perd in ["U", "us"]:
            train.index = pd.PeriodIndex(train.index, freq="U")
            using_p = 'U'
        elif perd == "N":
            train.index = pd.PeriodIndex(train.index, freq="N")
            using_p = 'N'
        return train, using_p
    def fit(self, y_panel_df):
        """For each time series fit each model in models.
        y_panel_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds', 'y']
        """

        fitted_models = []
        uids = []
        name_models = []
        for ts, meta_model in product(y_panel_df.groupby('unique_id'), self.models.items()):
            uid, y = ts
            name_model, model = deepcopy(meta_model)
            if 'R' != name_model[0]: # 非R语言版本
            # if name_model in ['Naive','Naive1','Naive3','Theta','ExponentialSmoothing','ETS','AutoETS','Croston']:
                y = y.set_index('ds')['y']
                y1, _ = self.sktime_add_freq(y.copy())
                try:
                    fitted_model = model.fit(y1)
                except Exception as e:
                    print('ljj debugging e',e)
                    from sktime.forecasting.naive import NaiveForecaster
                    forecaster = NaiveForecaster(strategy="drift")
                    fitted_model = forecaster.fit(y1)
                    # fitted_model = dask.delayed(forecaster.fit)(y1)

                # fitted_model.predict(fh=np.arange(14)) ###debugging
            else: ##### 默认R语言的包，需要这个流程
                y = y['y'].values
                # fitted_model = dask.delayed(model.fit)(y)
                fitted_model = model.fit(y)
            fitted_models.append(fitted_model)
            uids.append(uid)
            name_models.append(name_model)

        # fitted_models = dask.delayed(fitted_models).compute(scheduler=self.scheduler)
        fitted_models = pd.DataFrame.from_dict({'unique_id': uids,
                                                'model': name_models,
                                                'fitted_model': fitted_models})

        self.fitted_models_ = fitted_models.set_index(['unique_id', 'model'])
        return self

    def predict(self, y_hat_df):
        """Predict each model for each time series.
        y_hat_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds']
        """
        check_is_fitted(self, 'fitted_models_')

        y_hat_df = deepcopy(y_hat_df[['unique_id', 'ds']])

        forecasts = []
        uids = []
        dss = []
        name_models = []
        for ts, name_model in product(y_hat_df.groupby('unique_id'), self.models.keys()):
            uid, df = ts
            h = len(df)
            model = self.fitted_models_.loc[(uid, name_model)]
            model = model.item()
            # y_hat = dask.delayed(model.predict)(h)
            # if name_model in ['Naive','Naive1','Naive3','Theta','ExponentialSmoothing','ETS','AutoETS','Croston']:
            if 'R' != name_model[0]: # 非R语言版本
                try:
                    y_hat = model.predict(fh=np.arange(h+1)).values[1:]
                    # y_hat = dask.delayed(model.predict)(np.arange(h+1))

                except Exception as e:
                    print('ljj predict e',e)
                    y_hat = np.array([0.0]*h)
            else:##### 默认R语言的包，需要这个流程
                y_hat = model.predict(h)
                # y_hat = dask.delayed(model.predict)(h)
            forecasts.append(y_hat[-h:])
            uids.append(np.repeat(uid, h))
            dss.append(df['ds'])
            name_models.append(np.repeat(name_model, h))

        # forecasts = dask.delayed(forecasts).compute(scheduler=self.scheduler)
        forecasts = zip(uids, dss, name_models, forecasts)

        forecasts_df = []
        for uid, ds, name_model, forecast in forecasts:
            # print('dealing with',uid,name_model)
            dict_df = {'unique_id': uid,
                       'ds': ds,
                       'model': name_model,
                       'forecast': forecast}
            # df = dask.delayed(pd.DataFrame.from_dict)(dict_df)
            df = pd.DataFrame.from_dict(dict_df)
            forecasts_df.append(df)

        # forecasts = dask.delayed(forecasts_df).compute()
        forecasts = pd.concat(forecasts_df)

        forecasts = forecasts.set_index(['unique_id', 'ds', 'model']).unstack()
        forecasts = forecasts.droplevel(0, 1).reset_index()
        forecasts.columns.name = ''

        return forecasts

################################################################################
########## UTILS FOR FFORMA FLOW
###############################################################################

def temp_holdout(y_panel_df, val_periods):
    """Splits the data in train and validation sets.

    Parameters
    ----------
    y_panel_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']

    Returns
    -------
    Tuple
        - train: pandas df
        - val: pandas df
    """
    val = y_panel_df.groupby('unique_id').tail(val_periods)
    train = y_panel_df.groupby('unique_id').apply(lambda df: df.head(-val_periods)).reset_index(drop=True)

    return train, val

def calc_errors(y_panel_df, y_insample_df, seasonality, benchmark_model='Naive2'):
    """Calculates OWA of each time series
    usign benchmark_model as benchmark.

    Parameters
    ----------
    y_panel_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
    y_insample_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
        Train set.
    seasonality: int
        Frequency of the time seires.
    benchmark_model: str
        Column name of the benchmark model.

    Returns
    -------
    Pandas DataFrame
        OWA errors for each time series and each model.
    """

    assert benchmark_model in y_panel_df.columns

    y_panel = y_panel_df[['unique_id', 'ds', 'y']]
    y_hat_panel_fun = lambda model_name: y_panel_df[['unique_id', 'ds', model_name]].rename(columns={model_name: 'y_hat'})

    model_names = set(y_panel_df.columns) - set(y_panel.columns)

    errors_smape = y_panel[['unique_id']].drop_duplicates().reset_index(drop=True)
    errors_mase = errors_smape.copy()

    for model_name in model_names:
        errors_smape[model_name] = None
        errors_mase[model_name] = None
        y_hat_panel = y_hat_panel_fun(model_name)

        errors_smape[model_name] = evaluate_panel(y_panel, y_hat_panel, smape)
        errors_mase[model_name] = evaluate_panel(y_panel, y_hat_panel, mase, y_insample_df, seasonality)

    mean_smape_benchmark = errors_smape[benchmark_model].mean()
    mean_mase_benchmark = errors_mase[benchmark_model].mean()

    errors_smape = errors_smape.drop(columns=benchmark_model).set_index('unique_id')
    errors_mase = errors_mase.drop(columns=benchmark_model).set_index('unique_id')

    errors = errors_smape/mean_mase_benchmark + errors_mase/mean_smape_benchmark
    errors = 0.5*errors
    errors = errors

    return errors

def get_prediction_panel(y_panel_df, h, freq):
    """Construct panel to use with
    predict method.
    """
    df = y_panel_df[['unique_id', 'ds']].groupby('unique_id').max().reset_index()

    predict_panel = []
    for idx, df in df.groupby('unique_id'):
        date = df['ds'].values.item()
        unique_id = df['unique_id'].values.item()

        date_range = pd.date_range(date, periods=4, freq='D')
        df_ds = pd.DataFrame.from_dict({'ds': date_range})
        df_ds['unique_id'] = unique_id
        predict_panel.append(df_ds[['unique_id', 'ds']])

    predict_panel = pd.concat(predict_panel)

    return predict_panel
