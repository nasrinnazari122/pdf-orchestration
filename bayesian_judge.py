import numpy as np

class BayesianJudge:
    def __init__(self, alpha=2.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta

    def reliability(self, p):
        mean = self.alpha / (self.alpha + self.beta)
        return mean * p + (1 - mean) * 0.5
