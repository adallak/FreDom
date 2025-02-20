## Version 1.0.2 01/09/2022
import numpy as np
from scipy import signal
import math
from postprocess import threshold_till_dag

class InputError(ValueError):
    def __str__(self):
        return 'One of the which_freq and freqlist is allowed'    

class FreDom:
    """
    This class estimates topological ordering
    and recovers summary DAG of observed time series.
    """
    def __init__(self, X, lmbd=0.1, which_freq = None, nfreq=None,
                 detrend='constant', dfreq=1, freqlist = None,
                 thresh=None, rhoflex=True,
                 rho=1000, niter=100, alpha=1.8, 
                 abstol=1e-5, 
                 reltol=1e-4, diag=False,
                 L_init = None, provide_isdm = False,
                 sdm_halflength = None, provide_coh = False,
                 rho_max = 1e+6, verbose = False
                 ):
        # Data parameters
        self.X = X
        self.p = X.shape[1]
        self.N = X.shape[0]
        # Threshold parameters
        self.thresh = thresh
        self.rho_max = rho_max
        self.which_freq = which_freq
        self.type = "simple"
        self.niter = niter
      #  if (which_freq != None) and (freqlist != None):
      #      raise InputError("Only one of the which_freq and freqlist is allowed")
        assert which_freq != None or freqlist != None,\
        "Only one of the which_freq and freqlist should be provided"
        if (which_freq == None):
            if (nfreq is None) and (freqlist is None):
                self.nfreq = 10
                self.freqlist = list(range(1, self.nfreq))
            elif freqlist is None and nfreq is not None:
                self.nfreq = nfreq
                self.freqlist = list(range(1, self.nfreq))
            elif freqlist is not None:
                self.nfreq = len(freqlist)
                self.freqlist = [int(np.floor(x * self.N)) for x in
                    freqlist]
            self.M =  self.nfreq
        # Optimization parameters
            self.rhoflex = rhoflex
            self.rho = rho
            self.niter = niter
            self.alpha = alpha

        # Spectral density paremeters

        if sdm_halflength == None:
            self.halfwindow = math.floor(math.sqrt(self.N))
        else:
            self.halfwindow = sdm_halflength

        self.K = 2 * self.halfwindow + 1
        if which_freq is not None:
            if self.N %2 != 0:
                n_freq = int(np.floor(self.N/2))
            else:
                n_freq = int(self.N/2) - 1
            assert self.which_freq < 1/2,\
               "input which_freq should be less than 1/2"
            self.which_freq = int(np.floor(self.which_freq * self.N))
            self.lmbd = lmbd / self.K
        else:
            self.lmbd = lmbd /(self.K * self.M)
        #self.loweradj = np.zeros((self.p, self.p))
        #self.adj = np.zeros((self.p, self.p))
 
        self.abstol = abstol
        self.reltol = reltol
        self.diag = diag
        self.L_init = L_init
        self.provide = provide_isdm
        self.provide_coh = provide_coh
        self.verbose = verbose


    def fit(self, lmbd = None):
        self.get_order()
        self.get_structure(lmbd = lmbd)
        self.postprocess()
        if self.provide != False:
            if self.which_freq != None:
                self.theta =  self.L.conj().T @ \
                    self.L
            else:
                self.theta =  np.zeros(self.p * 
                    self.p * self.M).reshape((self.p, 
                    self.p, self.M)) + 0j
                for freq in range(self.M):
                    self.theta[:,:,freq] = self.L[:,:,freq].conj().T @ \
                        self.L[:,:, freq]

    def postprocess(self):
        perm = np.zeros((self.p, self.p)).astype(int)
        perm[np.arange(self.p), self.order.astype(int)] = 1
        if self.which_freq != None:
            self.L =  np.transpose(perm) @ self.L @ perm
            self.B = -np.copy(self.L)
            np.fill_diagonal(self.B, 0.0) 
        else:
            self.Z = np.transpose(perm) @ self.Z @ perm
            for iter in range(self.M):
                self.L[:,:, iter] =  np.transpose(perm) @ self.L[:,:, iter] * perm
            self.B = -np.copy(self.Z)
            np.fill_diagonal(self.B, 0.0) 


    def get_structure(self, lmbd = None):
        if lmbd != None:
            self.lmbd = lmbd
        if self.which_freq != None:
            self.updateL_j()
            #self.c_scsc_j()
        else:
            self.c_scsc()
    
    def get_sxx(self, window = None):
        if window != None:
            self.window = window
        self.myspecden()

    def get_xfft_j(self, freq):
        left = max( freq - self.halfwindow,0)
        right = min(self.N, self.halfwindow + freq) + 1
        return(self.xfft[left:right, :])

    def get_order(self):
        self.simple_sdm()
        self.orderEst()

    def simple_sdm(self):
        if type(self.X[1,1]) == np.complex128:
            xfft = np.fft.fftn(a = self.X, axes = [0])
            xfft_freq = np.fft.fftfreq(self.N) * self.N 
        else:
            xfft = np.fft.rfftn(a = self.X, axes = [0])
            xfft_freq = np.fft.fftfreq(self.N) * self.N 
        freq_ind = np.where(xfft_freq > 0)
        n_index = freq_ind[0].shape[0]
        if (self.which_freq == None):
            self.cglasso_ind = np.random.randint(1, n_index, size= self.nfreq)
            self.buffer = np.zeros((self.M, self.p))
            self.M = np.shape(self.cglasso_ind)[0]
        self.sxx = np.zeros(self.p * self.p * n_index).reshape((self.p,
         self.p, n_index)) + 0j
        self.coh = np.copy(self.sxx)
        xfft = xfft[freq_ind]
        self.xfft = xfft
        #N_half = xfft.shape[0] - 1
        #xfft = xfft[1:N_half, :]
        for i in range(self.p):
                for j in range(self.p):
                    self.sxx[i,j,:] = xfft[:,i] * np.conj(xfft[:,j]) / n_index
        kernel = 1 / self.K * np.ones(self.K)
        for i in range(self.p):
            for j in range(self.p):
                # _, tmp_coh = signal.coherence(self.X[:, i], 
                #                     self.X[:, j],
                #                     fs=self.dfreq, 
                #                     window = self.window,
                #                     nperseg= 2 * self.M,
                #                     detrend=self.detrend)
                self.sxx[i,j,:] = np.convolve(self.sxx[i,j,:], kernel, mode = "same")
        if self.provide_coh is True:    
            for i in range(self.p):
                for j in range(self.p):       
                    self.coh[i, j, :] = np.abs(self.sxx[i,j, :])**2 / \
                        np.real(self.sxx[i,i,:] * self.sxx[j,j,:])
                #tmp_coh[1:(self.M + 1)]

    def orderEst(self):
        """
        Estimates topological order
        of variables by extending
        Chen et. al (2019) approach
        """
        if (self.which_freq != None):
            self.orderEst_j(self.which_freq)
        else:
            for (j,freq) in enumerate(self.freqlist):
                done = None
                done = self.p
                S = self.sxx[:, :, freq].view(np.complex128)
                Sinv = np.linalg.inv(S)
                for i in range(self.p):
                    varmap = np.setdiff1d(np.arange(self.p), done)
                    v = np.argmin(np.real(np.diag(np.linalg.inv(
                        Sinv[np.ix_(varmap, varmap)]
                    ))))
                    done = np.append(done, varmap[v])
                self.buffer[j, :] = done[1:]
            uniq, cnt = np.unique(self.buffer, axis = 0, return_counts=True)
            self.order = uniq[np.argmax(cnt)].astype(int)

    def orderEst_j(self, freq):
        done = None
        done = self.p
        S = self.sxx[:, :, freq].view(np.complex128)
        try:
            ind = np.eye(self.p)
            eps = 1e-4
            Sinv = np.linalg.inv(S + eps * ind)
        except np.linalg.LinAlgError as err:
            ind = np.eye(self.p)
            eps = 1e-4
            Sinv = np.linalg.inv(S + eps * ind) 
        for i in range(self.p):
            varmap = np.setdiff1d(np.arange(self.p), done)
            v = np.argmin(np.real(np.diag(np.linalg.inv(
                Sinv[np.ix_(varmap, varmap)]
                ))))
            done = np.append(done, varmap[v])
        self.order = done[1:]

    def c_scsc(self):
        """
        This function recovers DAG
        given the topological ordering
        """
        self.geninit()
        tau_iner = tau_decr = 2
        mu = 10
        for (init,freq) in enumerate(self.freqlist):
            self.L[:,:, init] = np.diag(1 / np.sqrt(
                np.diag(self.sxx[:,:,freq])
            ))

        for k in range(self.niter):
            Zold = np.copy(self.Z)
            ## Update L
                
            for (iter_f,freq) in  enumerate(self.freqlist):
                sxx_f = self.sxx[:,:, freq]
                ordered_sxx_f = sxx_f[np.ix_(self.order, self.order)]
                
                self.L[:,:, iter_f] = self.updateL(ordered_sxx_f,
                self.Z, self.U[:,:, iter_f],
                self.L[:,:,iter_f])
                
                self.L[:,:, iter_f] = self.alpha * self.L[:,:, iter_f] +\
                                 (1- self.alpha) * Zold
            ## Update global Z 
            self.Z = self.updateZ()

            ## Update U
            for iter_f in  range(self.M):
                ## Update U
                self.U[:,:, iter_f] = self.U[:,:, iter_f] + (
                    self.L[:,:, iter_f] - self.Z
                )

                self.history_rnorm[k] = self.history_rnorm[k] + np.linalg.norm(
                    self.L[:,:,iter_f] - self.Z)


                self.history_epspri[k] = self.history_epspri[k] + np.maximum(
                    np.linalg.norm(self.L[:,:,iter_f]),
                    np.linalg.norm(self.Z)
                )
                self.history_snorm[k] = self.history_snorm[k]  + self.rho * np.linalg.norm(
                   self.Z - Zold)
                self.history_epsdual[k] = self.history_epsdual[k] + np.linalg.norm(
                    np.abs(self.U[:,:, iter_f])
                )
            self.history_objval[k] = self.objective()
            self.history_epspri[k] = self.M * self.p * self.abstol + (
                self.reltol * self.history_epspri[k]
            )
            self.history_epsdual[k] = self.M * self.p * self.abstol + (
                self.reltol * self.history_epsdual[k]
            )

            if (self.history_rnorm[k] < self.history_epspri[k]
                and self.history_snorm[k] < self.history_epsdual[k]):
                break

            if self.rhoflex == True:
                if self.rho < self.rho_max:
                        self.rho *= 1.35
                # if self.history_rnorm[k] > mu * self.history_snorm[k]:
                #     self.rho = tau_iner * self.rho
                #     self.U = self.U / tau_iner
                # elif self.history_rnorm[k] * mu < self.history_snorm[k]:
                #     self.rho = self.rho / tau_decr
                #     self.U = self.U * tau_decr
                # else:
                #     self.rho = np.copy(self.rho)

        if self.thresh != None:
            for f in range(self.M):
                self.L[:,:,f] = self.threshold(self.L[:,:,f])
            
        # adj_tens = np.copy(np.real(self.Z))
        # adj_tens[np.abs(adj_tens) != 0] = 1
        # for i in range(1,self.p):
        #     for j in range(i):
        #         if all(adj_tens[i, j,:] == 0):
        #             self.loweradj[i,j] = 0
        #         else:
        #             self.loweradj[i,j] = 1
        # perm = np.eye(self.p)
        # perm = perm[self.order, :]
        # self.adj = np.transpose(perm) @ self.adj @ perm
        self.history_objval = self.history_objval[self.history_objval != 0]
    


    def objective(self):
        """
        Estimated objective function
        """
        sum_var = 0
        for (iter_f,freq) in enumerate(self.freqlist):
            omega = np.transpose(
                np.conj(self.L[:,:, iter_f])) @ self.L[:,:, iter_f]
            mtx2 = self.sxx[:, :, freq]
            sum_var = sum_var + (np.real(np.trace(
                omega @ mtx2)) - np.log(np.real(np.linalg.det(omega))))
        
        penalty1 =  self.lmbd * (np.sum(np.abs(self.Z)) - np.real(np.sum(np.diag(self.Z))))
        for iter in range(self.M):
            penalty2 = self.rho * (np.linalg.norm(self.L[:,:,iter] - self.Z + self.U[:,:,iter])**2 - 
            np.linalg.norm(self.U[:,:, iter])**2)
        obj = self.K * sum_var + penalty1 + penalty2
        return obj

    def updateZ(self):
        """
        Updates W step in ADMM
        """
        A = np.sum(self.L + self.U, axis = 2) ##!! Self U is not defined yet
        try:   
            eps = 1e-8
            weight = np.maximum(1 - self.lmbd/ (self.rho * (np.abs(A) + eps)), 0)
        except RuntimeError:
            pass
        if self.diag == False:
            np.fill_diagonal(weight,1)
        return A * weight/ self.M
                   
    def updateL(self, S_f, Z_f, U_f, old_L_f):
        """
        Updates L step in ADDM
        """
        L = np.zeros((self.p, self.p)) + 0j
        L[0, 0] = 1 / np.sqrt(S_f[0,0])
        Y = Z_f - U_f
        for i in range(1, self.p):
            beta = old_L_f[i, :(i+1)]
            L[i, :(i+1)] = self.row_update(beta,
            S_f[:(i+1),:(i+1)], Y, i)
            #np.conj(S_f[:(i+1),:(i+1)]), Y, i)
        return L
# self.updateL(ordered_sxx_f,
#                self.Z, self.U[:,:, iter_f],
#                self.L[:,:,iter_f])
    def row_update(self, beta, S, Y, i):
        """
        Updates one row of tensor.
        """
        converge = False
        iter = 1
        S_rowi = np.copy(S[i, :])
        Y_i = np.copy(Y[i, :])
        while iter < self.niter and converge == False:
            old_beta = np.copy(beta)
            S_coli = np.copy(S[:, i])
            # Select indexes except i
            temp_index = np.setdiff1d(np.arange(len(beta)), i) 
            col_sum = np.sum(np.real(
                beta[temp_index] * S_coli[temp_index]
            ))
            beta[i] = (-col_sum + np.sqrt(
                col_sum**2 + 4 * S[i,i]
            )) / (2 * S[i,i])
            for j in range(i):
                S_colj = np.copy(S[:, j])
                temp_index = np.setdiff1d(np.arange(len(beta)), j)
                col_sum = np.sum(beta[temp_index]
                * S_colj[temp_index])
                beta[j] = (self.rho * Y_i[j] 
                - self.K * col_sum) / (self.K * S[j,j] + self.rho)
            if np.max(np.abs(old_beta - beta)) < self.abstol:
                converge = True
            else:
                iter = iter + 1
        return beta

    def threshold(self, L):
        """
        Thresholds the estimated matrix
        """
        L[np.abs(np.real(L)) <= 
        self.thresh] = 0
        return L

    def geninit(self):
        if type(self.L_init) == None.__class__:
            self.L =  np.zeros(self.p * 
            self.p * self.M).reshape((self.p, 
            self.p, self.M)) + 0j
            for (iter,freq) in enumerate(self.freqlist):
                np.fill_diagonal(self.L[:,:, iter],1 / np.diag(self.sxx[:,:,freq]))
        else:
            assert self.L_init.shape[0] == self.p
            assert self.L_init.shape[1] == self.p
            assert self.L_init.shape[2] == self.M
            self.L = self.L_init
        self.Z =  np.zeros(self.p * 
            self.p).reshape((self.p, 
            self.p)) + 0j
        self.U =  np.zeros(self.p *
            self.p * self.M).reshape((self.p, 
            self.p, self.M)) + 0j
        self.history_objval = np.zeros(self.niter)
        self.history_rnorm = np.zeros(self.niter)
        self.history_snorm = np.zeros(self.niter)
        self.history_epspri = np.zeros(self.niter)
        self.history_epsdual = np.zeros(self.niter)

    ### Implements classo

    def c_scsc_j(self):
        X = self.get_xfft_j(self.which_freq)
        self.L = np.zeros((self.p, self.p))*1j
        for i in range(1, self.p):
            self.L[i,:i] = np.squeeze(self.classo(X[:,i], X[:,:i]),1)

    def classo(self, Y, X, beta = None):
        self.n = Y.shape[0]
        Y = Y.reshape((self.n,-1))
        X = X.reshape((self.n, -1))
        p = X.shape[1]
        if beta is None:
            beta1 = np.zeros(p)
            beta2 = np.zeros(p)
            beta = beta1 + beta2 * 1j
        else:
            beta1 = np.real(beta)
            beta2 = np.imag(beta)
        U = self.returnU(X)
        W = self.returnW(Y)
        ex_U = self.extendU(U)
        gamma = self.returngamma(beta1, beta2)
        U_gamma = self.returnU_Gamma(ex_U, gamma)
        resid = W - self.returnSum(U_gamma)
        converge = False
        count = 0
        while converge == False:
            gamma_old = gamma[:]
            for j in range(p):
                gamma[j], resid, U_gamma[j] = \
                self.classoUpdate(j, ex_U, U_gamma, resid)
            error = np.sqrt(np.sum([(gamma[i] - gamma_old[i])**2 for i in range(p)]))
            count = count + 1
            if error  < self.abstol:
                converge = True
        beta1 = np.array([gamma[i][0] for i in range(p)]).reshape((p,1))
        beta2 = np.array([gamma[i][1] for i in range(p)]).reshape((p,1))
        beta = beta1 + beta2 * 1j
        return(beta)

    def classoUpdate(self, j, ex_U, U_gamma, resid):
        r_j = resid + U_gamma[j].reshape((-1,1))
        U_j = ex_U[j]
        gamma_j = np.squeeze(self.softthresh(U_j, r_j),1)
        U_gamma_j = U_j @ gamma_j
        resid = r_j - U_gamma_j.reshape((-1,1))
        return(gamma_j, resid, U_gamma_j)

    def softthresh(self, Uj, rj):
        Uj_rj = Uj.T @ rj
        n = Uj.shape[0] / 2
        eps = 1e-6
        thresh = np.maximum(1 - n * self.lmbd/(
                                np.sqrt(np.sum(Uj_rj**2))), 0)
        return(thresh * Uj_rj/ (np.linalg.norm(Uj,"f")**2))
    

    def returnU(self, X):
        N, p = X.shape
        U = np.zeros((2*N ,2*p))
        U[:N,:p] = np.real(X)
        U[N:,:p] = np.imag(X)
        U[:N,p:] = -np.imag(X)
        U[N:,p:] = np.real(X)

        return(U)
    
    def returnW(self, Y):
        N = Y.shape[0]
        W = np.zeros((2*N,1))
        W[:N] = np.real(Y).reshape((N,1))
        W[N:] = np.imag(Y).reshape((N,1))

        return(W)
    

    def extendU(self, U):
        p = int(U.shape[1]/ 2)
        return([U[:,[j,j+p]] for j in range(p)])
    
    def returngamma(self, beta1, beta2):
        p = beta1.shape[0]
        return([np.array([beta1[j], beta2[j]]) for j in range(p)])
    
    def returnU_Gamma(self, ext_U, gamma):
        p = len(gamma)
        return([ext_U[j]@ gamma[j] for j in range(p)])

    def returnSum(self, mat, j = None):
        if j is None:
            p = mat[0].shape[0]
            leng = len(mat)
            stor = np.zeros((p, 1))
            for i in range(leng):
                stor = stor + mat[i].reshape((-1,1))
            return(stor)
        else:
            p = mat[0].shape[0]
            mat = [mat[i] for i in range(p) if i != j]
            return(np.sum(mat))
    
## Implement FreDom for each frequency

    def updateL_j(self, initL = None):
        """
        Updates L 
        """
        S_f = self.sxx[:,:,self.which_freq]
        if (initL is None): 
            initL = np.zeros_like(S_f)
            eps = 1e-4
            np.fill_diagonal(initL, np.diag(S_f) + eps )
        self.L = np.zeros((self.p, self.p))*1j
        self.L[0,0] = 1 / np.sqrt(S_f[0,0])
        for i in range(1, self.p):
            beta = initL[i, :(i+1)]
            self.L[i, :(i+1)] = self.row_update_L(beta,
            S_f[:(i+1),:(i+1)],  i)
            #np.conj(S_f[:(i+1),:(i+1)]), Y, i)

# self.updateL(ordered_sxx_f,
#                self.Z, self.U[:,:, iter_f],
#                self.L[:,:,iter_f])
    def row_update_L(self, beta, S, i):
        """
        Updates one row of tensor.
        """
        converge = False
        iter = 1
        S_rowi = np.copy(S[i, :])
        while converge == False:
            old_beta = np.copy(beta)
            S_rowi = np.copy(S[i, :])
            # Select indexes except i
            temp_index = np.setdiff1d(np.arange(len(beta)), i) 
            row_sum = np.sum(np.real(
                beta[temp_index] * S_rowi[temp_index]
            ))
            beta[i] = (-row_sum + np.sqrt(
                row_sum**2 + 4 * S[i,i]
            )) / (2 * S[i,i])
            for j in range(i):
                S_rowj = np.copy(S[j, :])
                temp_index = np.setdiff1d(np.arange(len(beta)), j)
                row_sum = np.sum(beta[temp_index]
                * S_rowj[temp_index])
                beta[j] = self.softthresh_freq(row_sum, S[j,j])
         #   print(f"beta at row {i} is", np.round(np.real(beta),2))
            if np.max(np.abs(old_beta - beta)) < self.abstol:
                converge = True
            else:
                if (iter > self.niter):
                    err = np.max(np.abs(old_beta - beta))
                    converge = True
                    if (self.verbose is True):
                        print(f"Algorithm does not converge at error {err}")
                iter = iter + 1
        return beta
    
    def softthresh_freq(self, col_sum, Sii):
        thresh = np.maximum(1 -self.lmbd/(
                                2 * np.abs(col_sum)), 0)
        return(-thresh * col_sum/ (Sii))

    def myspecden(self):
        """
        Wrapper for spectral
        density esitmation function
        """
        for i in range(self.p):
            for j in range(self.p):
                self.freq, tmp = signal.csd(self.X[:, i],
                                                self.X[:,j], 
                                                fs=self.dfreq,
                                                window = self.window,
                                                nperseg= self.nfreq, 
                                                detrend=self.detrend,
                                                average = self.csd,
                                                noverlap = self.overlap,
                                                scaling = self.scaling)
                self.sxx[i, j, :] = tmp[1:(self.M + 1)]
                _, tmp_coh = signal.coherence(self.X[:, i], 
                                    self.X[:, j],
                                    fs=self.dfreq, 
                                    window = self.window,
                                    nperseg= self.nfreq,
                                    detrend=self.detrend)
                self.coh[i, j, :] = tmp_coh[1:(self.M + 1)]
        



class eBICFreDom(FreDom):
    """
    This class implements extended BIC
    """
    def __init__(self, X, lamlist = None, nfreq=10,
                 detrend='constant', dfreq=1,
                 thresh=None, rhoflex=True,
                 rho=1000, niter=100, 
                 alpha=1.8, nlam = 40,  
                 abstol=1e-5, reltol=1e-4, flmax = 1, 
                 diag=False, gamma = 0.5, flmin = 0.01,
                 stop_thresh = 1e-7, stopping = None):

        super().__init__(X = X, 
                nfreq = nfreq,
                detrend =  detrend, 
                dfreq  = dfreq, L_init = None,
                thresh = thresh, rhoflex = rhoflex,
                rho = rho, niter = niter, 
                alpha = alpha, abstol = abstol,
                reltol = reltol, diag = diag)
        
        if lamlist == None:
            self.myspecden()
            self.nlam = nlam
            self.flmin = flmin
            self.flmax = flmax
            self.pathgen()
        else:
            self.lamlist = np.sort(lamlist)[::-1].tolist()
            self.nlam = len(self.lamlist)
     
        eBIC_store = np.zeros(self.nlam)
        self.gamma = gamma
        self.stop_thresh = stop_thresh
        #self.X = X
        for i, lmbd in enumerate(self.lamlist) :
            self.fit(lmbd = lmbd)
            self.L_init = self.L
            eBIC_store[i] = self.eBICobjective()
            if stopping != None:
                if (i > 4 and 
                    (eBIC_store[i] - eBIC_store[i - 1]) < self.stop_thresh and
                    (eBIC_store[i-1] - eBIC_store[i - 2]) < self.stop_thresh and
                    (eBIC_store[i-2] - eBIC_store[i - 3]) < self.stop_thresh):
                    break
        self.eBIC_store = eBIC_store[np.abs(eBIC_store) > 0]
        self.eBIC_store = self.eBIC_store[::-1]
        self.lamlist = self.lamlist[::-1]
        self.eBIC_lmbd = self.lamlist[np.argmin(self.eBIC_store)] 


    def eBICobjective(self):    
        sum_var = 0
        for iter_f in range(self.M):
            omega = np.transpose(
                np.conj(self.L[:,:, iter_f])) @ self.L[:,:, iter_f]
            mtx2 = self.sxx[:, :, iter_f]
            nz = np.real(np.sum(self.L[:,:, iter_f]) - self.p)
            sum_var = sum_var + (2 * (np.real(np.trace(
                mtx2 @ omega)) - np.log(np.real(np.linalg.det(omega))))
            + np.log(2 * self.M*self.K) * nz + 4 * nz * self.gamma * self.p)
            #sum_var = sum_var + (2 * self.K * (np.real(np.trace(
            #    mtx2 @ omega)) - np.log(np.real(np.linalg.det(omega))))
            #+ np.log(2 * self.K * self.M) * nz + 4 * nz * self.gamma * self.p)
        obj = sum_var
        return obj


    def lammax(self):

        sighat = np.zeros((self.p - 1, self.p - 1))
        for i in range(1, self.p) :
            for j in range(0, self.p - 1) :
                if i != j:

                    sighat[i - 1, j] = np.max(np.real(self.sxx[i,j, :])) / \
                        np.sqrt(np.max(np.real(self.sxx[i,i,:])))

        self.lam_max = np.asscalar(np.max(sighat)) * self.flmax

    
    def pathgen(self):
        self.lammax()
        lam_min = self.lam_max * self.flmin
        self.lamlist = np.exp(np.linspace(np.log(self.lam_max), 
        np.log(lam_min), 
        self.nlam)).tolist()


    


class select_thresh(FreDom):
    def __init__(self, X, lmbd=0.1, which_freq = None, nfreq=None,
                 detrend='constant', dfreq=1,
                 thresh=None, rhoflex=True,
                 rho=1000, niter=100, alpha=1.8, 
                 abstol=1e-5, 
                 reltol=1e-4, diag=False,
                 L_init = None, provide_isdm = False,
                 sdm_halflength = None, provide_coh = False,
                 rho_max = 1e+6, verbose = False,
                  n_thr = 15, min_thr = 0.001, gamma = 0.25,
                max_thr =  None, thr_list = None):               
        super().__init__(X =X, lmbd=lmbd, which_freq = which_freq, 
                       nfreq=nfreq, detrend=detrend, dfreq=dfreq, 
                       thresh=thresh, rhoflex=rhoflex,
                       rho=rho, niter=niter, alpha=alpha, abstol=abstol, 
                       reltol=reltol, diag=diag, L_init = L_init, 
                       provide_isdm = provide_isdm, 
                       sdm_halflength = sdm_halflength, 
                       provide_coh = provide_coh,
                       rho_max = rho_max, verbose = verbose)
        
        self.n_thr = n_thr
        self.min_thr = min_thr
        self.gamma = gamma
        if max_thr is None:
            self.max_thr = 5 * min_thr
        else:
            self.max_thr = max_thr
        self.thr_list = thr_list
        assert n_thr > 0,\
        "n_thr should be positive integer"
        assert min_thr > 0,\
        "min_thr should be positive number"
        assert max_thr > 0,\
        "max_thr should be positive number"
        assert min_thr < max_thr,\
        "min_thr should be greater than max.thr"
        if thr_list is not None:
            self.n_thr = len(self.thr_list)
        else:
            self.thr_list = np.linspace(self.min_thr, 
                                        self.max_thr, 
                                        num= self.n_thr)
        self.ebicscores = []

    def fit(self):
        super().fit()
        for (i,thr) in enumerate(self.thr_list):
            B_post = self.adj_postprocess(self.L, thr)
            edges = np.sum(np.abs(B_post)!=0) - self.p
            log_lik = self.likelihood(B_post)
            self.ebicscores.append(2 * log_lik +
                            edges * np.log(self.K) +
                            4 * self.gamma * edges * np.log(self.p))
        #      buffer.aic[i] = (2 * log.lik +
        #                          edges * rtsglasso$K)

        min_index = np.argmin(self.ebicscores)
        #  min_index.aic = which.min(buffer.aic)
        ebic_thr = self.thr_list[min_index]
        #  aic.thr = thr.list[min_index.aic]
        self.B = self.adj_postprocess(self.B, ebic_thr)
        nzero = int(np.sum(np.abs( self.B )==0) - self.p * (self.p + 1)/2)
        edges = int(np.sum(np.abs( self.B ) != 0)) 
        print("Selected threshold level is %1.4f"% ebic_thr)
        print(f"Number of zeros are equal {nzero}")
        print(f"Number of edges are equal {edges}")

    def likelihood(self, B_post):
        freq = self.which_freq
        sdm   = self.sxx[:,:, freq]
        norm  = np.sum(abs(B_post)) - np.sum(np.real(np.diagonal(B_post)))
        det = np.linalg.det(B_post) #np.prod(np.diagonal(B_post)[0])
        #print(f"Determinant is {det}")
        K = self.K
        return(K*(np.sum(np.real(np.diagonal(sdm @ np.conj(B_post).T
                                              @B_post))) -2* np.real(np.log(det)))) 

    def adj_postprocess(self,B, threshold=0.3):
        """Post-process estimated solution:
            (1) Thresholding.
            (2) Remove the edges with smallest absolute weight until a DAG
                is obtained.
        Args:
            B (numpy.ndarray): [d, d] weighted matrix.
            graph_thres (float): Threshold for weighted matrix. Default: 0.3.
        Returns:
            numpy.ndarray: [d, d] weighted matrix of DAG.
        """
        B_nodiag = np.copy(B)
        np.fill_diagonal(B_nodiag, 0)
        B_nodiag[np.abs(B_nodiag) <= threshold] = 0    # Thresholding
        B_nodiag, _ = threshold_till_dag(B_nodiag)
        np.fill_diagonal(B_nodiag, np.diagonal(B))

        return B_nodiag   
    
    def adj_postprocess_inv(self,B, threshold=0.3):
        """Post-process estimated solution:
            (1) Thresholding.
            (2) Remove the edges with smallest absolute weight until a DAG
                is obtained.
        Args:
            B (numpy.ndarray): [d, d] weighted matrix.
            graph_thres (float): Threshold for weighted matrix. Default: 0.3.
        Returns:
            numpy.ndarray: [d, d] weighted matrix of DAG.
        """
        nfreq = B.shape[2]
        B_nodiag = np.empty_like(B)
        for i in range(nfreq):
            B_nodi = np.copy(B[:,:,i])
            np.fill_diagonal(B_nodi, 0)
            B_nodi[np.abs(B_nodi) <= threshold] = 0    # Thresholding
            B_nodiag[:,:,i], _ = threshold_till_dag(B_nodiag)
            np.fill_diagonal(B_nodiag[:,:,i], np.diagonal(B[:,:,i]))

        return B_nodiag   
    
