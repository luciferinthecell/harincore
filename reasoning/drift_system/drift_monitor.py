import numpy as np

class DriftMonitor:
    def __init__(self, window:int=6, threshold:float=0.20):
        self.window=window
        self.threshold=threshold
        self.series=[]
    
    def push(self, rhythm_score:float):
        self.series.append(rhythm_score)
        recent=self.series[-self.window:]
        if len(recent)<3:
            return {"status":"insufficient"}
        stdev=float(np.std(recent))
        mean=float(np.mean(recent))
        drift=stdev>self.threshold
        return {"stdev":round(stdev,3),"mean":round(mean,3),"drift":drift,"window":len(recent)} 