import numpy as np
import math
import cmath
import random
import pdb
from scipy import signal
import scipy.stats as stats
from numpy import ndarray
from typing import List
import igraph as ig
##########################################
### For SID calculation 
### uncomment lines 16 - 24 and 126,129 
##########################################

# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr

# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr

# sid = importr('SID')
# from rpy2.robjects import numpy2ri
# numpy2ri.activate()



class GenComplexDAG:


    def __init__(self, p, T, nedges = None, prob = None, 
       freqlist = None, provide_dag = None,
       causalorderlist = None,
       invariance = False,
       seed = 10034):
        
        self.p = p
        self.prob = prob
        if (nedges is None):
            self.nedges = 15
        else:
            self.nedges = nedges
        self.T = T
        self.seed = seed
        if invariance is True and causalorderlist is not None:
            if len(causalorderlist) > 1:
                print("The first element of causalorderlist is selected")
        if freqlist is not None and causalorderlist is not None:
            assert len(freqlist) == len(causalorderlist)
        if (freqlist is None):
            self.freqlist = [1/10, 1/5, 2/5]
        else:
            self.freqlist = freqlist
        assert np.max(self.freqlist) < 1/2,\
        "The maximum frequency in freqlist should be less than 0.5"
        self.causalorderlist = causalorderlist
        self.invariance = invariance
        self.set_seed(self.seed)
        if provide_dag is not None:
            self.DAG = provide_dag["dag"]
            self.ord = provide_dag["ord"]
        else:
            self.genDAG()
        self.genData()


    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed) 

    def genDAG(self):
        """
        Generates stritly lower trinagular weighted
        adjacency matrix B
        Args:  
        """
        if self.invariance is False:
            self.DAG = np.zeros((self.p, self.p, len(self.freqlist))) 
            self.ord = []
            for (j,_) in enumerate(self.freqlist):
                ord, DAG = self.simulate_dag(self.p, self.nedges, 
                                                    "ER")    

                self.DAG[:,:,j] = DAG
                self.ord.append(ord)       
            # for (j,_) in enumerate(self.freqlist):
            #     DAG = np.zeros((self.p, self.p))
            #     if (self.causalorderlist is None):
            #         causalOrder = np.random.choice(np.arange(self.p), 
            # size = self.p, replace = False)
            #     else:
            #         causalOrder = self.causalorderlist[j]
            #     for i in range(self.p - 2):
            #         node = causalOrder[i]
            #         potParent = causalOrder[(i+1) : self.p]
            #         self.set_seed(self.seed * (i + 1))
            #         numParent = np.sum(np.random.binomial(n = 1,
            #         p = self.prob, size = (self.p - (i + 1))))
            #         self.set_seed(self.seed * (i + 1))
            #         Parent =  np.random.choice(potParent, 
            #         size = numParent, replace = False)
            #         Parent =  np.append(causalOrder[i + 1], Parent)
            #         DAG[node, Parent] = 1
            #     node = causalOrder[self.p - 2]
            #     self.set_seed(self.seed)
            #     Parent = np.random.binomial(n = 1, size = 1,
            #     p = self.prob)
            #     DAG[node, causalOrder[self.p - 1]] = 1
            #     self.DAG[:,:,j] = DAG
            #     self.ord.append(np.flip(causalOrder))

        else:         
            # DAG = np.zeros((self.p, self.p))
            # self.set_seed(self.seed)
            # causalOrder = np.random.choice(np.arange(self.p), 
            # size = self.p, replace = False)
            # for i in range(self.p - 2):
            #     node = causalOrder[i]
            #     potParent = causalOrder[(i+1) : self.p]
            #     self.set_seed(self.seed * (i + 1))
            #     numParent = np.sum(np.random.binomial(n = 1,
            #     p = self.prob, size = (self.p - (i + 1))))
            #     self.set_seed(self.seed * (i + 1))
            #     Parent =  np.random.choice(potParent, 
            #     size = numParent, replace = False)
            #     Parent =  np.append(causalOrder[i + 1], Parent)
            #     DAG[node, Parent] = 1
            # node = causalOrder[self.p - 2]
            # self.set_seed(self.seed)
            # Parent = np.random.binomial(n = 1, size = 1,
            # p = self.prob)
            # DAG[node, causalOrder[self.p - 1]] = 1
            # self.DAG = DAG
            # self.ord = np.flip(causalOrder) 
            self.ord, self.DAG = self.simulate_dag(self.p, self.nedges, 
                                                   "ER")

    def simulate_dag(self, d, s0, graph_type):
        """Simulate random DAG with some expected number of edges.

        Args:
            d (int): num of nodes
            s0 (int): expected num of edges
            graph_type (str): ER, SF, BP

        Returns:
            B (np.ndarray): [d, d] binary adj matrix of DAG
        """
        def _random_permutation(M):
            # np.random.permutation permutes first axis only
            order = np.random.permutation(M.shape[0])
            P = np.eye(M.shape[0])[order,:]
            return (order,P.T @ M @ P)

        def _random_acyclic_orientation(B_und):
            return np.tril(_random_permutation(B_und)[1], k=-1)

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        if graph_type == 'ER':
            # Erdos-Renyi
            G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)
        elif graph_type == 'SF':
            # Scale-free, Barabasi-Albert
            G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
            B = _graph_to_adjmat(G)
        elif graph_type == 'BP':
            # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
            top = int(0.2 * d)
            G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
            B = _graph_to_adjmat(G)
        else:
            raise ValueError('unknown graph type')
        order, B_perm = _random_permutation(B)
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
        return (order,B_perm)

    def complexB(self):
        self.B = np.zeros((self.p, self.p, len(self.freqlist))) *1j
        Bmin = 1
        #aug_freqlist = [1 - x for x in self.freqlist]
        #aug_freqlist.sort()
        #self.freqlist.extend(aug_freqlist)
        if self.invariance is False:
            for (i,omega) in enumerate(self.freqlist):
                if (omega <= 0.5):
                    DAG = self.DAG[:,:,i]
                else:
                    DAG = self.DAG[:,:,i - int(len(self.freqlist) / 2)]
                B = np.array(DAG, copy = True, dtype= complex)
                size = np.sum(DAG).astype(int)
                first = np.random.uniform(low = Bmin+2,
                    high = 5, size = size) *  math.cos(4*math.pi * omega)
                second = np.random.uniform(low = Bmin, 
                    high = 3, size = size) * 1.2j * math.sin(2*math.pi * omega)
                B[B == 1] =  first + second
                self.B[:,:,i] = np.round(B,2)
                #self.B[:,:,i] = self.B[:,:,i][np.ix_(self.ord, self.ord)]
        else:
            for (i,omega) in enumerate(self.freqlist):
                B = np.array(self.DAG, copy = True, dtype= complex)
                size = np.sum(self.DAG).astype(int)
                first = np.random.uniform(low = Bmin+2,
                    high = 5, size = size) *  math.cos(4*math.pi * omega)
                second = np.random.uniform(low = Bmin, 
                    high = 3, size = size) * 1.2j * math.sin(2*math.pi * omega)
                B[B == 1] =  first + second
                self.B[:,:,i]  = np.round(B,2)
                #self.B[:,:,i] = self.B[:,:,i][np.ix_(self.ord, self.ord)]
   
    def genR_f(self):
        self.complexB()
        total_len =  2*len(self.freqlist)
        self.genR_f = np.zeros((self.p, self.T, total_len)) * 1j
        for (i,omega) in enumerate(self.freqlist):
            if (self.freqlist[i] == 0.5 or self.freqlist[i] == 1 ):
                Z =  np.random.normal(loc = 0, 
                    scale = 1/np.sqrt(self.T), size = (self.T, self.p))
            else:
                Z = (np.random.normal(loc = 0, 
                    scale = 1/(2 *np.sqrt(self.T)), size = (self.T,self.p
                    )) + 1j * \
                       np.random.normal(loc = 0, 
                    scale = 1/(2 *np.sqrt(self.T)), size = (self.T ,self.p
                    )))
            self.genR_f[:,:,i] = np.linalg.solve(np.eye(self.p) - 
            self.B[:,:,i],Z.T)
            self.genR_f[:,:,total_len - i - 1] = np.conj(self.genR_f[:,:,i])



    def genData(self):
        burn = 200
        X = np.zeros((self.p, self.T))
        self.genR_f()
        total_len =  2*len(self.freqlist)
        for t in range(self.T):
            for (i,omega) in enumerate(self.freqlist):
                X[:,t ] = X[:,t] +  np.real(self.genR_f[:,t, i] * np.exp(2j * np.pi *
                                                          omega * (t+1)) +\
                                     self.genR_f[:,t, total_len - i - 1] * \
                                        np.exp(2j * np.pi *
                                                          (1-omega) * (t+1))) 
                
            
        self.X = X.T



class SimResult():
    """
    This class measures performance
    of the proposed method
    """
    def __init__(self, est_dag, est_ord, 
    true_dag, true_ord ):

        self.true_ord = true_ord.astype(int)
        self.true_adj = true_dag[np.ix_(self.true_ord, self.true_ord)]
        self.est_ord  = est_ord.astype(int)
        self.est_adj  = est_dag[np.ix_(self.est_ord, self.est_ord)]
        self.p = self.est_adj.shape[0]
      #  sid_res = sid.structIntervDist(self.true_adj, self.est_adj)
        self.result = {}
        self.result["SHD"] = self.hammingDistance().astype(int)
      #  self.result["SID"] = np.asscalar(np.array(sid_res[1]))
        self.result["Kendall tau"] = np.round(self.kendall(),2)
        self.result["recall"] = np.round(self.recall(),2)
        self.result["precision"] = np.round(self.precision(),2)

        
    def hammingDistance(self):
        hammingDis = np.sum(abs(self.est_adj - self.true_adj))
        hammingDis -= 0.5 * np.sum(self.est_adj * 
        np.transpose(self.est_adj) * (1 - self.true_adj) * 
        np.transpose(1 - self.true_adj) +
                                  self.true_adj * 
                                  np.transpose(self.true_adj) *
                                   (1 - self.est_adj) * 
                                   np.transpose(1 - self.est_adj))
                        
        return hammingDis
    
    def kendall(self):
        tau, _ = stats.kendalltau(self.true_ord, self.est_ord)
        return tau
    
    def recall(self):
        if np.sum(self.true_adj) > 0:
            return (np.sum(self.true_adj * self.est_adj) /
             np.sum(self.true_adj))
        else:
            return 0

        
    def precision(self):
        if np.sum(self.est_adj) > 0:
            return (np.sum(self.true_adj * self.est_adj) / 
            np.sum(self.est_adj) )
        
    def fdr(self):
        if np.sum(self.est_adj) > 0:
            return (1 - np.sum(self.true_adj * self.est_adj) / 
            np.sum(self.est_adj))
        else:
            return 0


class GenDataFromSVAR():
    def __init__(self, T = 1000, 
                    seed =11245,
                    k = 5):
        self.T     = T
        self.seed = seed
        self.k = k


    def genDAG(self):
        """
        Generates stritly lower trinagular weighted
        adjacency matrix B
        Args:  
        """
        DAG = np.zeros((self.k, self.k))
        np.random.seed(self.seed)
        causalOrder = np.random.choice(np.arange(self.k), 
        size = self.k, replace = False)
        for i in range(self.k - 2):
            node = causalOrder[i]
            potParent = causalOrder[(i+1) : self.k]
            np.random.seed(self.seed * (i + 1))
            numParent = np.sum(np.random.binomial(n = 1, 
            p = self.prob, size = (self.k - (i + 1))))
            np.random.seed(self.seed * (i + 1))
            Parent =  np.random.choice(potParent, 
            size = numParent, replace = False)
            Parent =  np.append(causalOrder[i + 1], Parent)
            DAG[node, Parent] = 1
        node = causalOrder[self.k - 2]
        np.random.seed(self.seed)
        Parent = np.random.binomial(n = 1, size = 1,
         p = self.prob)
        DAG[node, causalOrder[self.k - 1]] = 1
        self.DAG = DAG
        self.ord = np.flip(causalOrder) 

    def exp2(self):
        burn = 500
        self.k = 4 #A.shape[0]
        self.B0 = np.array([[0,0,0,0],
                            [0,0,0,0],
                            [1,1,0,0],
                            [0,1,0,0]])
        self.B1 = np.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,1,1,0],
                            [0,0,1,1]])    
        self.DAG = self.B0 + self.B1 
        np.fill_diagonal(self.DAG,0) 
        self.DAG[self.DAG > 1] = 1
        self.ord = np.array([0,1,2,3])             
        np.random.seed(self.seed)
        errs = np.random.normal(size = self.k * 
        (burn + self.T + 1)).reshape(self.k, burn + self.T + 1)
        sizeB0 = np.sum(self.B0).astype(int)
        sizeB1 = np.sum(self.B1).astype(int)
        self.B0 = self.B0.astype(float)
        self.B1 = self.B1.astype(float)
        self.B0[self.B0 == 1] = np.random.uniform(low = 0.2,
         high = 0.8, size = sizeB0 ) * (2 *
          np.random.binomial(n = 1,
          p = 0.5, size = sizeB0) - 1)
        self.B1[self.B1 == 1] = np.random.uniform(low = 0.2,
         high = 0.8, size = sizeB1) * (2 * 
         np.random.binomial(n = 1,
          p = 0.5, size = sizeB1) - 1)
       #self.B1  =  (np.eye(self.k) - self.B0) @ A
        self.X = np.zeros((self.k, burn + self.T + 1))
        self.X[:, 0] = errs[:, 0]
        for i in range(1, burn + self.T + 1):
            self.X[:, i] = np.linalg.solve(np.eye(self.k) - 
            self.B0, self.B1 @ self.X[:, i - 1] + errs[:, i])
        self.X = np.transpose(self.X[:, burn:])


    def exp3(self):
        burn = 500
        self.k = 3 #A.shape[0]
        self.B0 = np.array([[0,0,0],
                            [1,0,0],
                            [0,1,0]])
        self.B1 = np.array([[1,0,0],
                            [1,1,0],
                            [0,1,1]])    
        self.DAG = self.B0 + self.B1  
        np.fill_diagonal(self.DAG,0) 
        self.X = np.zeros((burn + self.T + 1, self.k))
        self.X[0,:] = np.random.normal(scale = 0.4, size = self.k)
        self.ord = np.array([0,1,2])             
        np.random.seed(self.seed)
        err = np.random.normal(size = self.k)
        for t in range(1, burn + self.T + 1):
            err = np.random.normal(scale = 0.5, size = self.k) + 0.1
            # np.random.uniform(low = -0.5,
            # high = 0.5, size = 4)
            self.X[t, 0] = 0.8*self.X[t-1,0] + 0.6 * err[0]
            self.X[t, 1] = -0.4 * self.X[t-1, 1] + np.cos(
                self.X[t-1,1] - 1) + 0.6 * err[1]
            self.X[t, 2] = -0.4 * self.X[t-1,2] + 0.5 * (
                self.X[t-1,1]) + \
                np.sin(self.X[t-1,1]) + 0.6 * err[2]
        self.X = self.X[burn:, :]
        self.X = (self.X[0:, :])


    # def linSVAR(self, prob):
    #     burn = 500
    #     self.prob = prob
    #     # self.A = A
    #     self.k = 5 #A.shape[0]
    #     Bmin = 0.3
    #     #self.genDAG()
    #     self.genDAG()
    #     self.B0 = np.copy(self.DAG)
    #     np.random.seed(self.seed)
    #     errs = 0.2 * np.random.normal(size = self.k * 
    #     (burn + self.T + 1)).reshape(self.k, burn + self.T + 1)
    #     size = np.sum(self.B0).astype(int)
    #     self.B0[self.B0 == 1] = np.random.uniform(low = Bmin,
    #      high = 1, size = size ) * (2 * np.random.binomial(n = 1,
    #       p = 0.5, size = size) - 1)
    #     self.B1 = np.array([[0, 1, 1, 0 ,0],
    #                         [0, 1, 1, 0, 0],
    #                         [0, 0, 1, 1, 0],
    #                         [1, 0, 0, 1, 0],
    #                         [0, 0, 0, 1, 1]])
    #     #self.B1  =  (np.eye(self.k) - self.B0) @ A
    #     self.X = np.zeros((self.k, burn + self.T + 1))
    #     self.X[:, 0] = errs[:, 0]
    #     for i in range(1, self.T + 1):
    #         self.X[:, i] = np.linalg.solve(np.eye(self.k) - 
    #         self.B0, self.B1 @ self.X[:, i - 1] + errs[:, i])
    #     self.summarygraph = self.B0 + self.B1
    #     np.fill_diagonal(self.summarygraph,0)
    #     self.X = np.transpose(self.X[:, burn:])

    def nonlinSVAR(self):
        np.random.seed(self.seed)
        self.k = 4
        burn = 500
        self.X = np.zeros((burn + self.T + 1, self.k))
        self.X[0,:] = (0.2*np.random.normal(size = self.k)
        ).reshape((1, self.k))
        A_coef = np.random.uniform(low = -0.8,
         high = 0.8, size = 12 ) #* (2 * np.random.binomial(n = 1,
          #p = 0.5, size = 11) - 1)
        print(A_coef)
        for t in range(1, burn + self.T + 1):
                err =  np.random.normal(size = 4)
                self.X[t, 3] = A_coef[0] * self.X[t,2]**2 + \
                    A_coef[1] * self.X[t - 1, 3] + A_coef[9] * self.X[t-1,1] +  err[0]
                self.X[t, 0] = A_coef[2] * self.X[t,1]**2 + A_coef[3]  \
                    * self.X[t-1, 0] + A_coef[4] * self.X[t-1,1]**2 + err[1]
                self.X[t, 2] = A_coef[5] * self.X[t,0]**3 + A_coef[6]  \
                    * self.X[t-1,1]**2 + A_coef[7] * self.X[t - 1,2] + err[2]
                self.X[t, 1] = A_coef[8] * self.X[t - 1, 1] + err[3]
        self.X = self.X[burn:, :]
        self.DAG = np.array([[0, 1, 0, 0],
                             [0, 0, 0, 0],
                             [1, 1, 0, 0],
                             [0, 1, 1, 0]])
        self.B0 = np.array([[0, 1, 0, 0],
                             [0, 0, 0, 0],
                             [1, 0, 0, 0],
                             [0, 1, 0, 0]])
        self.B1 =  np.array([[1, 1, 0, 0],
                             [0, 1, 0, 0],
                             [0, 1, 1, 0],
                             [0, 1, 0, 1]])
        self.ord = np.array([1, 0, 2, 3]).astype(int)


    def linDAGSVAR(self):
        burn = 500
        Bmin = 0.3
        #self.genDAG()
        #self.genDAG()
        self.k = 5
        #self.B0 = np.copy(self.DAG)
        # np.random.seed(self.seed)
        # errs = (0.2 * np.random.normal(size = self.k * 
        # (self.T + 1))**3).reshape(self.k, self.T + 1)
        # size = np.sum(self.B0).astype(int)
        # self.B0[self.B0 == 1] = np.random.uniform(low = Bmin,
        #  high = 1, size = size ) * (2 * np.random.binomial(n = 1,
        #   p = 0.5, size = size) - 1)
        self.B0 = np.array([[0, 1, 0, 0 ,0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0]])
        self.B1 = np.array([[0, 1, 1, 0 ,0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0]])
        np.random.seed(self.seed)
        errs = (np.random.normal(size = self.k * 
            (burn + self.T + 1))).reshape(self.k, burn + self.T + 1)
        size0 = np.sum(self.B0).astype(int)
        size1 = np.sum(self.B1).astype(int)
        self.B0 = self.B0.astype(float)
        self.B1 = self.B1.astype(float)
        self.B0[self.B0 == 1] = np.random.uniform(low = Bmin,
            high = 1, size = size0 ) * (2 * np.random.binomial(n = 1,
            p = 0.5, size = size0) - 1)
        self.B1[self.B1 == 1] = np.random.uniform(low = Bmin,
            high = 1, size = size1 ) * (2 * np.random.binomial(n = 1,
            p = 0.5, size = size1) - 1) 
        A = np.linalg.solve(np.eye(self.k) - self.B0, self.B1)
        eigens = np.linalg.eigvals(A)
        if (np.max(eigens) > 1):
            A = A / (np.linalg.norm(A, ord = 2)
                + 0.01)
            self.B1 = (np.eye(self.k) - self.B0) @ A 
        self.X = np.zeros((self.k, burn + self.T + 1))
        self.X[:, 0] = errs[:, 0]
        for i in range(1, burn + self.T + 1):
            self.X[:, i] = np.linalg.solve(np.eye(self.k) - 
            self.B0, self.B1 @ self.X[:, i - 1] + errs[:, i])

        self.X = np.transpose(self.X[:, burn:])
        self.DAG = np.array([[0, 1, 1, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0]])
                    
        self.ord = np.array([2, 1, 0, 3, 4]).astype(int)


def baseline(p, prob = 0.5, order = None):
        DAG = np.zeros((p, p))
        if (order is None):   
            causalOrder = np.random.choice(np.arange(p), 
            size = p, replace = False)
        else:
            causalOrder = np.flip(order)
        for i in range(p - 2):
            node = causalOrder[i]
            potParent = causalOrder[(i+1) : p]
            numParent = np.sum(np.random.binomial(n = 1, p = prob, 
            size = (p - (i + 1))))
            Parent =  np.random.choice(potParent, size = numParent, 
            replace = False)
            Parent =  np.append(causalOrder[i + 1], Parent)
            DAG[node, Parent] = 1
        node = causalOrder[p - 2]
        Parent = np.random.binomial(n = 1, size = 1, p = prob)
        DAG[node, causalOrder[p - 1]] = 1
        DAG = DAG
        ord = np.flip(causalOrder) 
        return ord, DAG


class gendatafromVAR:

    def __init__(self, K: int,
                 p: int, 
                 T: int, 
                 sigma: ndarray,
                 burn: int,
                 type: str = "block",
                 block: int = 2,
                 band: int = 3):
    
        if type == "block":
            assert K % block == 0 , \
                """
                K should be multiple of block
                """
        elif type == "chain":
            assert band < K,\
                """
                Bandwidth is too large
                """
        else:
            raise ValueError("type should be chain or block \n")

        assert sigma.shape[0] == sigma.shape[1] \
            == K, \
                """
                Size of sigma is incompatible
                """

        self.K = K
        self.p = p
        self.T = T
        self.block = block
        self.sigma = sigma
        self.burn = burn
        self.A = np.zeros((K, K * p))
        self.type = type
        self.band = band

    def get_coeffmat(self):
        if self.type == "block":
            self.gen_blockmat()
        elif self.type == "chain":
            self.gen_chainmat()




    def gen_blockmat(self):
        block_size = int(self.K / self.block)
        for lag in range(self.p):
            left = lag * self.K 
            right = (lag + 1) * self.K 
            for bl in range(self.block):

                left1 = bl * block_size
                right1 = (bl + 1) * block_size 
                self.A[:,left:right][left1:right1, left1:right1] =\
                np.random.uniform(low = -0.5, high = 0.5, 
                size = block_size**2) .reshape((block_size,
                block_size))
                # if bl != (self.block - 1):
                #     self.A[:,left:right][left1:(right1 + block_size), 
                #     (right1 - 2):right1] =\
                #         np.random.uniform(low = -0.5, high = 0.5, 
                #         size = 2 * block_size * 2).reshape((2 * block_size,2))
                #     self.A[:,left:right][right1 - 1, left1:(right1 + block_size)] =\
                #         np.random.uniform(low = -0.5, high = 0.5, 
                #         size = block_size * 2).reshape((1, 2 * block_size))   
                #self.A[:,left:right] = np.tril(self.A[:,left:right], k = 0)    
                self.A = np.round(self.A, 3)
        
        
        isCausal, companion = self.companion(self.A)
        while isCausal == False:
           self.A = self.A / (np.linalg.norm(companion, ord = 2)
            + 0.01)
           isCausal, companion = self.companion(self.A)

    def gen_chainmat(self):
        for lag in range(self.p):
            left = lag * self.K 
            right = (lag + 1) * self.K 
            self.A[:,left:right] = self.gen_bandmat()

        isCausal, companion = self.companion(self.A)
        while isCausal == False:
            self.A = np.round(self.A / (np.linalg.norm(companion, ord = 2)
             + 0.01),3)
            isCausal, companion = self.companion(self.A)



    def gen_bandmat(self):
        matrix = np.zeros((self.K,self.K))
        k = self.band
        for k in range(-k, (k+1)):
            if k < 0:
                fill = np.round(np.random.uniform(low = -1, high = 1,
                    size = self.K + k),3)
                np.fill_diagonal(matrix[-k:, :k], fill)
            if k > 0:
                fill = np.round(np.random.uniform(low = -1, high = 1,
                    size = self.K - k),3)
                np.fill_diagonal(matrix[:-k, k:], fill)
            if k == 0:
                fill = np.round(np.random.uniform(low = -1, high = 1,
                    size = self.K),3)
                np.fill_diagonal(matrix, fill)

        return matrix

    def get_data(self):
        inteVect = np.zeros((self.K, 1))
        assert np.min(np.linalg.eigvals(self.sigma)) > 0 , \
            """
            Covariance matrix is not positive definite
            """
        self.X = np.empty((self.K, self.T + self.burn))
        self.X[:] = np.nan
        cholsigma = np.linalg.cholesky(self.sigma)
        self.X[:,0: self.p] = np.random.normal(size = self.K * self.p
        ).reshape((self.K, self.p))
        for i in range(self.p, self.T + self.burn):
            Z = np.random.normal(size = self.K).reshape((self.K, 1))
            error = cholsigma @ Z
            X_fl = np.flip(self.X[:, (i - self.p): (i)].flatten('F'
            ).reshape((-1,1)))
#            X_fl = np.flip(X_fl)
            self.X[:, i] = (inteVect + self.A @ X_fl + error).reshape(self.K)

        self.X = np.transpose(self.X[:, (self.burn): 
        (self.T + self.burn)])
        
    def fit(self):
        self.get_coeffmat()
        self.get_data()
        self.DAG = np.tril(self.A[:, : self.K], k = -1)
        self.DAG[np.abs(self.DAG) > 0] = 1
        self.order = np.arange(0, self.K)

    def companion(self, A: ndarray) -> List:
        K = A.shape[0]
        p = int(A.shape[1] / K)
        companion = np.zeros((K *p, K * p))
        companion[0:K, 0:(K * p)] = np.copy(A)
        if p > 1:
            companion[(K ):(K*p), 0:(K * p - K)] = \
            np.eye(K *p - K)
        eigens = np.linalg.eigvals(companion)
        isCausal = np.max(eigens) < 1

        return isCausal, companion


class gendatafromlargeSVAR(gendatafromVAR):
    def __init__(self, K: int,
             p: int, 
             T: int, sigma: ndarray,
             burn: int,
             type: str="block",
             block: int = 2,
             band: int = 3):
        super().__init__(K = K, p = p, block = block ,
        T = T, type = type, band = band,
        sigma = sigma, burn = burn)
    
    def B_0(self):
        if self.type == "chain":
            self.chain_B0()
        elif self.type == "block":
            self.block_B0()


    def block_B0(self):
        self.B_0 = np.zeros((self.K, self.K))
        block_size = int(self.K / self.block)
        for bl in range(self.block):

            left1 = bl * block_size
            right1 = (bl + 1) * block_size 
            self.B_0[left1:right1, left1:right1] =\
            np.random.uniform(low = -0.4, high = 0.4, 
                size = block_size**2).reshape((block_size,
                block_size))
            if bl != (self.block - 1):
                self.B_0[right1 - 1, left1:(right1 + block_size+ 1)] =\
                    np.random.uniform(low = -0.4, high = 0.4, 
                    size = block_size*2).reshape((1,
                    block_size * 2))
                self.B_0[left1:(right1 + block_size+ 1), right1 - 1] =\
                    np.random.uniform(low = -0.4, high = 0.4, 
                    size = block_size*2).reshape((1,
                    block_size * 2))
        self.B_0 = np.round(self.B_0,2)            
        self.B_0 = np.tril(self.B_0, k = -1)


    def chain_B0(self):
        self.B_0 = super().gen_bandmat()
        self.B_0 = np.tril(self.B_0, k = -1)


    def get_data(self):
        inteVect = np.zeros((self.K, 1))
        assert np.min(np.linalg.eigvals(self.sigma)) > 0 , \
            """
            Covariance matrix is not positive definite
            """
        self.X = np.empty((self.K, self.T + self.burn))
        self.X[:] = np.nan
        cholsigma = np.linalg.cholesky(self.sigma)
        self.X[:,0: self.p] = np.random.normal(size = self.K * self.p
        ).reshape((self.K, self.p))
        for i in range(self.p, self.T + self.burn):
            Z = np.random.normal(size = self.K).reshape((self.K, 1))
            error = cholsigma @ Z
            X_fl = np.flip(self.X[:, (i - self.p): (i)].flatten('F'
            ).reshape((-1,1)))
#            X_fl = np.flip(X_fl)
            self.X[:, i] = np.linalg.inv(np.eye(self.K) - self.B_0) @ \
                 (inteVect + self.A @ X_fl + error).reshape(self.K)

        self.X = np.transpose(self.X[:, (self.burn): 
        (self.T + self.burn)])

#    def get_data(self):
#        super().get_data()
#        self.X = self.X @ np.transpose(np.linalg.inv(np.eye(self.K) - self.B_0))

    def get_coeffmat(self):
        self.B_0()
        super().get_coeffmat()
        self.DAG = self.B_0[:]
        self.DAG[np.abs(self.DAG) > 0] = 1
        self.ord = np.arange(self.K)


    def fit(self):
        self.get_coeffmat()
        self.get_data()

    
"""
The rest of the code has been modified from:
https://github.com/xunzheng/notears/blob/master/notears/utils.py
"""
import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape).astype(np.complex64)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape) + \
            1j * np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n) + \
                1j*np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X.astype(np.complex64)
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(np.abs(W).tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d]) + 1j * np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        print(f"parents of {j} are {parents}")
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond) 
    fdr = float( len(false_pos)) / max(pred_size, 1) # len(reverse) +
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(false_pos)) / max(cond_neg_size, 1) # len(reverse) + 
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}



