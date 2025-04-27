class GreyWolf:
    def __init__(self):
        self.Position = set()  # Current seed set
        self.Cost = []         # [f1,f2,f3]
        self.GridIndex = None
        self.GridSubIndex = None