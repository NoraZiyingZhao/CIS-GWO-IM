class GreyWolf:
    def __init__(self):
        self.Position = set()  # Current seed set
        self.Cost = []         # [spread, cost, fairness]
        self.GridIndex = None
        self.GridSubIndex = None