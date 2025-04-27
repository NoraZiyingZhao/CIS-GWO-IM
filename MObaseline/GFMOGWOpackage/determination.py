from MObaseline.GFMOGWOpackage.GreyWolf import *
# from fitness_function import *

def determineDomination(self):
    # Note here we are taking particles as 1D-array
    npop = self.greyWolvesNum
    pop = self.greyWolves

    for i in range(npop):  # (0 .. greyWolvesNum -1)
        pop[i].dominated = False

        for j in range(i):
            if (pop[j].dominated == False):
                if (pop[i].dominates(pop[j])):
                    pop[j].dominated = True
                elif (pop[j].dominates(pop[i])):
                    pop[i].dominated = True
                    break

def getNonDominatedWolves(self):
    num_of_obj = self.greyWolvesNum
    pop = self.greyWolves

    num_of_obj = self.greyWolvesNum
    pop = self.greyWolves
    nd_pop = []
    for i in range(num_of_obj):
        if not pop[i].dominated:
            nd_pop.append(pop[i])
    return np.array(nd_pop)

def getCosts(self):
    pop = self.archive
    num_of_obj = len(pop)
    pop_cost = []
    for i in range(num_of_obj):
        pop_cost.append(pop[i].cost)
    pop_cost = np.array(pop_cost)
    costs = np.transpose(pop_cost)
    return costs
