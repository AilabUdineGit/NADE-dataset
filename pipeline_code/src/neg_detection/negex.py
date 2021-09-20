import pandas as pd
from spacy.training import  offsets_to_biluo_tags as biluo_tags_from_offsets
import spacy
from negspacy.negation import Negex
from utils.interval_merger import IntervalMerger

class NegExScope():

    def __init__(self, df, task = "negation", build = False, save_intermediate = True):
        self.smm4h19 = df
        self.task = task
        self.save_intermediate = save_intermediate
        self.nlp = spacy.load("en_core_sci_lg")
        self.err = 0
        self.nlp.add_pipe("negex", last=True)
    
    def biluo_to_iob(self,biluo):
        return [t.replace('U', 'B', 1).replace('L', 'I', 1) for t in biluo]
    
    def iob_to_binary(self,iob):
        return [1 if e[0]=="I" or e[0]=="B" else 0 for e in iob]

    def run_negex(self):
        intervals = list()
        iob_tags_list = list()
        overlpapper = IntervalMerger()
        for i,elem in self.smm4h19.iterrows():
            if type(elem.tweet) == str:
                doc = self.nlp(elem.tweet)
                annotations = overlpapper.merge([[e.start_char, e.end_char] for e in doc.ents if e._.negex])
                iob_tags = self.biluo_to_iob(biluo_tags_from_offsets(doc, [(s,e,"") for s,e in annotations]))
                intervals.append(annotations)
                iob_tags_list.append(iob_tags)
            else:
                self.err +=1 
        self.smm4h19['negation_intervals'] = intervals
        self.smm4h19['negation_scope_iob'] = iob_tags_list

        return self.smm4h19