import numpy as np


def plus(x):
    return 0 if x < 0 else x

def minus(x):
    return 0 if x > 0 else -x

def shock(x):
    return np.sqrt(x)


class BaseDemand:
    def __init__(self,
                 cost):
        self.cost = cost

    def profit(self, price):
        pass


class SimpleDemand(BaseDemand):
    def __init__(self, 
                 cost=0.2,
                 intercept=1,
                 slope=-2):
        self.cost = cost
        self.intercept = intercept
        self.slope = slope

    def demand(self, price):
        return plus(self.intercept + self.slope * price)

    def profit(self, price):
        return self.demand(price) * (price - self.cost)


class ReferenceDemand(BaseDemand):
    def __init__(self,
                 cost=0.2,
                 intercept=1,
                 slope=-2,
                 alpha=-1,
                 beta=0.3,
                 gamma=-0.1):
        self.cost = cost
        self.intercept = intercept
        self.slope = slope
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def demand(self, price_t, price_t_1):
        today_demand = self.intercept + self.slope * price_t
        reference_demand = self.alpha*shock(plus(price_t - price_t_1)) + self.beta*shock(minus(price_t - price_t_1))+\
                            self.gamma*np.sign(np.abs(price_t-price_t_1))
        return plus(today_demand + reference_demand)

    def profit(self, price_t, price_t_1):
        return self.demand(price_t, price_t_1) * (price_t - self.cost)


class CompetitionDemand(BaseDemand):
    def __init__(self,
                 cost=0.8,
                 intercept=16.4 ,
                 slope=-30,
                 alpha=-30,
                 beta=30,
                 gamma=-1):
        self.cost = cost
        self.intercept = intercept
        self.slope = slope
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def demand(self, price_t, com_t,price_t_1):
        today_demand = self.intercept + self.slope * price_t
        competition_demand = self.alpha*shock(plus(price_t - (com_t + 0.03))) + self.beta*shock(minus(price_t - com_t))
        reference_demand = self.gamma*np.sign(np.abs((price_t-price_t_1)))
        return plus(today_demand + competition_demand + reference_demand)

    def profit(self, price_t, com_t,price_t_1):
        return self.demand(price_t, com_t,price_t_1) * (price_t - self.cost)