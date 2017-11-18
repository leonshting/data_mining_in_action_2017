import numpy as np
from sklearn.base import BaseEstimator


SVM_PARAMS_DICT = {
    'C': 100,
    'random_state': 42,
    'iters': 1000,
    'batch_size': 10,
}


import numpy as np
from random import randint
import random


np.random.seed(42)
random.seed(42)


class MySVM(object):
    def __init__(self, C=10000, batch_size = 100, iters=10000, **kwargs):
        self.C = C # regularization constant
        self.batch_size = batch_size
        self.iters = iters

    # f(x) = <w,x> + w_0
    def f(self, x):
        return np.dot(self.w, x) + self.w0

    # a(x) = [f(x) > 0]
    def a(self, x):
        return 1 if self.f(x) > 0 else -1
    
    # predicting answers for X_test
    def predict(self, X_test):
        o_o = np.array([self.a(x) for x in X_test])
        o_o[o_o == -1] = 0
        return o_o

    # l2-regularizator
    def reg(self):
        return 1.0 * sum(self.w ** 2) / (2.0 * self.C)

    # l2-regularizator derivative
    def der_reg(self):
        return self.w/self.C

    # hinge loss vectorized
    def loss(self, x, answer):
        return np.vectorize(lambda x_v, answer_v: max([0, 1 - answer_v * self.f(x_v)]),
                            signature='(m),()->()')(x, answer)

    # hinge loss derivative
    def _dl(self, x_v, answer_v):
        return -answer_v if 1 - answer_v * self.f(x_v) > 0 else 0.0
    
    def der_loss(self, x, answer):
        return np.vectorize(lambda x_v, answer_v: self._dl(x_v, answer_v), signature=
                           '(m),()->()')(x, answer)
    
    def der_loss_wrt_w(self, x, answer):
        #print(self.der_loss(x, answer))
        return np.mean((np.multiply(x.T, self.der_loss(x, answer))), axis=1)
    
    def der_loss_wrt_w0(self, x, answer):
        return np.mean(self.der_loss(x, answer))

    def trans_to_01(self, y):
        y[y == -1] = 1
        return y
    def trans_to_11(self, y):
        y[y == 0] = -1 
        return y
    
    def get_params(self, *args, **kwargs):
        return {
            'C': self.C,
            'batch_size': self.batch_size,
            'iters': self.iters
        }
    
    # fitting w and w_0 with SGD
    def fit(self, X_train, y_train):
        dim = len(X_train[0])
        self.w = np.random.rand(dim) # initial value for w
        self.w0 = np.random.randn() # initial value for w_0
        
        y_train = self.trans_to_11(y_train)
        # 10000 steps is OK for this example
        # another variant is to continue iterations while error is still decreasing
        loss_a = 1.
        delta = 1.
        cnt = 0
        glob_cnt = 0
        #stops if too long
        while (cnt<100 or abs(delta/loss_a) > 1e-3) and glob_cnt < self.iters:  
            
            # random example choise
            # rand_index = randint(0, len(X_train) - 1,) # generating random index
            rand_index = np.random.randint(low=0, high=X_train.shape[0], size=self.batch_size)
            x = X_train[rand_index]
            y = y_train[rand_index]
            
            
            
            loss_b = self.loss(x, y).sum()

            # simple heuristic for step size
            
            step = 1./(glob_cnt+1)
            # w update
            #print(self.der_loss_wrt_w(x, y), self.der_reg())
            
            self.w += step * (-self.der_loss_wrt_w(x, y) - self.der_reg())
            
            # w_0 update
            self.w0 += -step * self.der_loss_wrt_w0(x, y)
            
            loss_a = self.loss(x, y).sum()
            delta = abs(loss_a - loss_b)
            if abs(delta/loss_a) > 1e-3: 
                cnt = 0
            else:
                cnt+=1
            glob_cnt += 1 
        return self