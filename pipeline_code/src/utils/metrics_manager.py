from utils.evaluator import Evaluator

class Metrics():
    def __init__(self, df):
        self.df = df

    def f1(self, precision, recall):
        return  2 * ((recall*precision)/((recall+precision) if (recall+precision) != 0 else 1)) 

    def return_metrics(self):
        evaluat = Evaluator([e for e in self.df.iob_correct.values],[e for e in self.df.iob_prediction.values],["ADR"])
        a,b = evaluat.evaluate()
        return a