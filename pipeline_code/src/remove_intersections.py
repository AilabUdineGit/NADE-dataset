import pandas as pd

class CreateTwoDataset():
    def __init__(self, df):
        self.df = df
    
    def check(self, elem, intervals):
        return sum([int(elem[0]) in range(int(s),int(e)) or int(elem[1])-1 in range(int(s),int(e)) for s,e in intervals if not (pd.isna(s) or pd.isna(e) )]) > 0

    def create(self, keep_intersection = False):
        df_baseline = self.df.apply(lambda row: self.build_baseline_df(row, keep_intersection),axis=1)
        return (self.df, df_baseline)

    def build_baseline_df(self,row, keep_intersection):
        row['all_predicted_intervals'] = row['predicted_intervals']
        row['predicted_intervals'] = [elem for elem in row.predicted_intervals if not self.check(elem, row.negation_intervals)]
        return row