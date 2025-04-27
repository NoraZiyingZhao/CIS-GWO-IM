import numpy as np

class GreyWolf:

    def __init__(self):
        # self.dim = dim
        self.position = []
        self.position_dict = []
        self.corresponded_nodes = []
        # self.position=np.zeros(self.dim,dtype=int)
        #self.velocity=[]
        self.cost=float("-inf") #fairness value is equal to -inf or not?
        self.dominated=False
        self.best = {}
        self.best['Position']=[]
        self.best['Cost']=[]
        self.best['corresponded_nodes'] = []
        self.gridIndex=[]
        self.gridSubIndex=[]

    def dominates(self,y):      # returns True if self dominates y, namely,when each cost of x is smaller, x dominates y, cost minimization problem
        x= self.cost
        y= y.cost
        #dom = all(x<=y) and any(x<y)
        flag = 1
        dom = 0
        for i in range(len(x)):
            if(x[i]<y[i]):
                flag = 0

        if(flag==1):
            for i in range(len(x)):
                if(x[i]>y[i]):
                    dom = 1
                    break
        return dom