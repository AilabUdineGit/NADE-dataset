from datetime import datetime
from icecream import ic
import os
import json
import pandas as pd
from config import *
from utils.metrics_manager import Metrics

def mic(to_print):
    if Config.LOG.value:
        ic(to_print)

def save_results(df, df_baseline,model,dataset_to_evaluate,folder_name):
    mic("Saving results...")
    if not os.path.exists(os.path.join(Config.PATH_OUTPUT.value, str(folder_name))):
        os.makedirs(os.path.join(Config.PATH_OUTPUT.value, str(folder_name)))
    date = datetime.now().strftime("[%d/%m/%Y_%H:%M:%S]")

    # save dataframe
    save_model_name = model.replace("/","_")
    df.to_pickle(os.path.join(Config.PATH_OUTPUT.value,folder_name, f"complete___{save_model_name}.pickle"))
    df_baseline.to_pickle(os.path.join(Config.PATH_OUTPUT.value,folder_name, f"cutted___{save_model_name}.pickle"))
    ####
    
    # save metrics
    if not os.path.exists(os.path.join(Config.PATH_OUTPUT.value, str(folder_name), "metrics.csv")):
        df_metrics = pd.DataFrame({})
    else:
        df_metrics = pd.read_pickle(os.path.join(Config.PATH_OUTPUT.value, str(folder_name), "metrics.pickle"))
    
    negade = df.loc[[s.replace("\n","") for s in open(os.path.join(Config.PATH_INPUT.value, "split_test", "sample_negade.txt")).readlines()]]
    noade = df.loc[[s.replace("\n","") for s in open(os.path.join(Config.PATH_INPUT.value, "split_test", "sample_noade.txt")).readlines()]]
    metrics_negade = Metrics(negade).return_metrics()
    metrics_noade = Metrics(noade).return_metrics()
    metrics_ade = Metrics(df.loc[set(df.index.values)-set(negade.index.values)-set(noade.index.values)]).return_metrics()
    
    b_negade = df_baseline.loc[[s.replace("\n","") for s in open(os.path.join(Config.PATH_INPUT.value, "split_test", "sample_negade.txt")).readlines()]]
    b_noade = df_baseline.loc[[s.replace("\n","") for s in open(os.path.join(Config.PATH_INPUT.value, "split_test", "sample_noade.txt")).readlines()]]
    b_metrics_negade = Metrics(b_negade).return_metrics()
    b_metrics_noade = Metrics(b_noade).return_metrics()
    b_metrics_ade = Metrics(df_baseline.loc[set(df_baseline.index.values)-set(b_negade.index.values)-set(b_noade.index.values)]).return_metrics()

    metrics_complete = Metrics(df).return_metrics()
    metrics_cutted = Metrics(df_baseline).return_metrics()

    df_metrics = df_metrics.append(pd.Series({
        'model': model,
        'dataset': dataset_to_evaluate,
        'metrics_cutted': metrics_cutted,#metrics_df,
        'metrics_complete': metrics_complete,#,
        'metrics_negade': metrics_negade,
        'metrics_noade': metrics_noade,
        'metrics_ade': metrics_ade,
        'b_metrics_negade': b_metrics_negade,
        'b_metrics_noade':  b_metrics_noade,
        'b_metrics_ade':    b_metrics_ade
        #'spurious_cutted': metrics_df['partial']['spurious'],
        #'spurious_complete': metrics_df_baseline['partial']['spurious']
    }),ignore_index=True)

    df_metrics.to_csv(os.path.join(Config.PATH_OUTPUT.value, str(folder_name), "metrics.csv"))
    df_metrics.to_pickle(os.path.join(Config.PATH_OUTPUT.value, str(folder_name), "metrics.pickle"))
    ###

    # save model info
    with open(os.path.join(Config.PATH_OUTPUT.value,folder_name,"config.json"),"w") as fp:
        json.dump({
            'date': date,
            'neg_model': Config.NEG_MODEL.value,
            'task': Config.TASK.value
        },fp)
    mic("Results saved in: {}!".format(os.path.join(Config.PATH_OUTPUT.value,folder_name)))
    ###

def get_folder_name():
    i = 0
    found = True
    while found:
        if os.path.exists(os.path.join(Config.PATH_OUTPUT.value, str(i))):
            i += 1
        else:
            found = False
    return str(i)


def get_evaluated_raw_dataset(path):
    df = pd.read_pickle(path)
    if "id" in df.columns:
        df = df.rename(columns={'id': "tweet_id", "text": "tweet"})
    df = df.set_index("tweet_id")
    return df


