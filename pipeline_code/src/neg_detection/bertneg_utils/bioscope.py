import pandas as pd
from config import *
import os
from utils.interval_merger import IntervalMerger
from utils.tagger import Tagger
from utils.utils import mic

class Bioscope():
    def __init__(self, task, train_size = 0.8):
        self._task = task
        self._train_size = train_size
        if task not in ['negation', 'speculation']:
            mic("Must be 'negation' or 'speculation'")
        self._sentences = pd.read_pickle(os.path.join(Config.BIOSCOPE.value, "bioscope_sentences.pickle"))
        all_scope       = pd.read_pickle(os.path.join(Config.BIOSCOPE.value, "bioscope_scopes.pickle"))

        # create neg
        scope_neg = all_scope[all_scope.type == task]
        scope_neg = scope_neg.apply(lambda row: self._create_row(row), axis=1).drop(columns=['document_id','sentence_id','dataset'])
        self._sentences = self._sentences.apply(lambda row: self._create_row(row), axis=1).drop(columns=['document_id','sentence_id','dataset'])
        
        df_neg = self._create_dataset(scope_neg,block=False)
        new_df = self._create_final(df_neg)
        df_neg = Tagger(new_df[['text_id','text']],new_df,"").get().rename(columns={'text_id': 'id', 'text': 'sentence', 'tokens': 'spacy_tokens'}).drop(columns=['biluo_tags', 'io_tags'])
        
        # create spec
        scope_spec = all_scope[all_scope.type != task]
        scope_spec = scope_spec.apply(lambda row: self._create_row(row), axis=1).drop(columns=['document_id','sentence_id','dataset']).sample(frac=1)
        scope_spec = scope_spec[~scope_spec.id.isin(df_neg.id.values)].sample(400)

        df_spec = self._create_dataset(scope_spec,block=True)
        
        new_df = self._create_final(df_spec)
 
        df_spec = Tagger(new_df[['text_id','text']],new_df,"").get().rename(columns={'text_id': 'id', 'text': 'sentence', 'tokens': 'spacy_tokens'}).drop(columns=['biluo_tags', 'io_tags'])
        df_spec = df_spec.apply(lambda row: self._clean(row), axis=1)
        
        #concat
        self._df = pd.concat([df_spec,df_neg])

        ## create final df
        new_df = pd.DataFrame({})
        keys = self._df.id.unique()
        for key in keys:
            tmp = self._df[self._df.id == key]
            new_df = new_df.append(pd.Series({
                "id": str(key),
                "spacy_tokens": tmp.spacy_tokens.values[0],
                "sentence": tmp.sentence.values[0],
                "iob_tags": tmp.iob_tags.values[0]
            }),ignore_index=True)            
        self._df = new_df

    def _clean(self,row):
        row.iob_tags = ["O"]*len(row.iob_tags)
        return row

    def _create_final(self,df):
        new_df = pd.DataFrame({})
        for idx, row in df.iterrows():
            for (s,e) in row.correct_intervals:
                new_df = new_df.append(pd.Series({
                    "text_id": row.id,
                    "text": row.sentence,
                    "begin": int(s),
                    "end": int(e),
                    "type": "ADR"
                }),ignore_index=True)
        return new_df

    def _create_dataset(self,scope,block):
        keys = scope.id.unique()
        df = pd.DataFrame({})
        for key in keys:
            tmp = scope[scope.id == key]
            df = df.append(pd.Series({
                "id": key,
                "sentence": self._sentences[self._sentences.id == key].sentence.values[0],
                "correct_intervals": IntervalMerger().merge([[s,e] for (s,e) in zip(tmp.start.values, tmp.end.values)])
            }), ignore_index=True)
        return df


    def _create_row(self, row):
        row['id'] = f"{row.dataset}@{row.document_id}@{row.sentence_id}"
        return row
    
    def get_train_val_sets(self):
        train_set_size = int(len(self._df)*self._train_size)
        self._df = self._df.sample(frac=1).reset_index(drop=True)
        t = self._df.loc[:train_set_size]
        v = self._df[~self._df.id.isin(t.id.values)]

        return t, v#self._df.loc[train_set_size+1:]

    def get_df(self):
        return self._df.sample(frac=1).reset_index(drop=True)
