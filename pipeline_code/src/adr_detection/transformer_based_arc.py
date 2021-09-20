from tqdm import tqdm
import pandas as pd
import numpy as np
from utils.utils import mic

class ModelSpans():
    def __init__(self):
        pass

    def from_raw_to_cleaned_data(self, raw_data_path, original_dataset):
        """
            Take the raw data (BERT predictions) with columns:
             - tweet_id
             - start
             - end 
             - token
            in a dataset with columns:
             - tweet_id
             - tweet
             - predicted_ade_intervals
        """
        df = pd.read_pickle(raw_data_path)
        keys = list(original_dataset.index.values)
        new_df = pd.DataFrame({})

        for key in keys:
            key = str(key)
            sentence = original_dataset.loc[key].tweet
            if key in df.tweet_id.values:
                tmp = df[df.tweet_id == key]
                preds = list(zip(tmp.start.values,tmp.end.values))
                sentence = tmp.tweet.values[0]
            else:
                preds = []

            new_df = new_df.append(pd.Series({
                'tweet_id': str(key),
                'tweet': sentence,
                'predicted_intervals': preds
            }), ignore_index=True)
        
        new_df = new_df.set_index("tweet_id")
        return new_df

    def evaluate(self,
                 raw_data_path,
                 task,
                 original_dataset,
                 no_gold_labels = True):
        original_dataset.index = original_dataset.index.map(str)
        df = self.from_raw_to_cleaned_data(raw_data_path, original_dataset)
        if no_gold_labels:
            df['correct_intervals'] = [[] for _ in range(len(df))]
        else:
            df = pd.concat([df, original_dataset[['correct_intervals']]], axis=1)
            df['correct_intervals'] = [k if k != np.nan else [] for k in df['correct_intervals'].values]
        df['type'] = task
        df['tweet_normalized'] = df.tweet
        return df