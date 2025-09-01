import numpy as np
import control as ct
import matplotlib.pyplot as plt

from tqdm import tqdm
from tqdm.notebook import tnrange

import pickle

from torch import Tensor
import torch
from torch.optim import AdamW,Adam,SGD,NAdam

from scipy.optimize import basinhopping
from scipy.optimize import minimize as sciminimize
from scipy.stats import chi2
from scipy.special import expit
import scipy

from .simu import *
from .model_base import *
from .torch_gradient_descent import *
import functools

def evaluatorSS(model,block_size,xs,theta,C_prev,ys,us,u_test,y_test):
    N = xs.shape[1]
    
    #print('true x',xs,'true u',us,'true y',ys, 'theta',theta)

    C = np.linalg.pinv(model.fim(torch.tensor(theta),torch.tensor(xs[:,[-block_size]]),torch.tensor(us[:,-block_size:]))[0].numpy()+np.linalg.pinv(C_prev))
        
    x_prop,C_prev = propogate_x(theta,xs[:,[-block_size]],model.transition_np,us[:,-block_size:],C)
        

    M = theta.shape[0]
    C_theta = C[:M,:M]
    tr_crb = np.abs(np.trace(C_theta@np.diag(1/(theta.flatten()**2))))
                  
    mse = 0
    x_ = xs[:,[0]]
    Hn = model.H.numpy()
    for i in range(u_test.shape[0]):
        mse += ((y_test[:,[i]]-Hn@x_)**2)
        x_ = model.transition_np(theta,x_,u_test[:,[i]])
                
    mse/= u_test.shape[0]
    return tr_crb,mse,C_prev  

def propogate_analys(theta,x_init,f,F,us,C):
    J_ = C.clone()
    x_ = x_init.clone()
    for i in range(us.shape[1]):
        jac_prev = F(theta,x_,us[:,[i]])
        J_ = jac_prev@J_@jac_prev.T
        x_ = f(theta,x_,us[:,[i]])
        
    return torch.vstack([theta,x_]).detach().numpy(),J_.detach().numpy()

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = scipy.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
    
def get_sigmoid_points(v,Pv,alpha,beta):
    dim = v.shape[0]
    Nx = 2*dim + 1
    kappa = 0#3-Nx#
    
    
    lambd = (alpha**2)*(dim+kappa) - dim
    radius = np.sqrt(dim+lambd)
    wm0 = lambd/(dim+lambd)
    wc0 = lambd/(dim+lambd) + (1-alpha**2+beta)
    wci = 0.5/(dim+lambd)
    
    vset = np.zeros((dim,Nx))
    vset[:,0] = v[:,0]
    if Pv.shape[0]>1:
        rv = radius*scipy.linalg.cholesky(nearestPD(Pv),lower=False) #cholesky?
    else:
        rv = radius*np.sqrt(Pv)
        
    for i in range(1,Nx):
        if (i-1)//dim==0:
            vset[:,i] = v[:,0] + rv[(i-1)%dim,:]
        else:
            vset[:,i] = v[:,0] - rv[(i-1)%dim,:]
            
    return vset,wm0,wc0,wci

def covariance_x_w(x,Px,w,Pw,f,u,alpha=1,beta=2):
    x_set,wmx0,wcx0,wcxi = get_sigmoid_points(x,Px,alpha,beta)#(x_dim,N)
    w_set,wmw0,wcw0,wcwi = get_sigmoid_points(w,Pw,alpha,beta)#(w_dim,M)
    dimx,N = x_set.shape
    dimw,M = w_set.shape
    x_pred_set = np.zeros((dimx,N*M))
    weights = np.zeros((N*M,1))
    for i in range(M):
        for j in range(N):
            x_pred_set[:,[i*N+j]] = f(w_set[:,[i]],x_set[:,[j]],u)
            if j==0:
                weightx = wmx0
            else:
                weightx = wcxi
                
            if i==0:
                weightw = wmw0
            else:
                weightw = wcwi
            
            weights[i*N+j] = weightx*weightw
    
    x_pred_mean = x_pred_set@weights
    
    Px_ = (x_pred_set-x_pred_mean)@np.diag(weights[:,0])@(x_pred_set-x_pred_mean).T
    return Px_

def covariance_x(x,Px,f,us,alpha=0.001,beta=2.5):
    
    x_set,wmx0,wcx0,wcxi = get_sigmoid_points(x,Px,alpha,beta)
    dimx,N = x_set.shape
    x_pred_set = np.zeros((dimx,N))
    weights_m = np.zeros((N,1))
    weights_c = np.zeros((N,1))
    for i in range(N):
        x_pred_set[:,[i]] = f(x_set[:,[i]],us)
        if i==0:
            weights_m[i] = wmx0
            weights_c[i] = wcx0
        else:
            weights_m[i] = wcxi
            weights_c[i] = wcxi
    x_pred_mean = x_pred_set@weights_m
    Px_ = (x_pred_set-x_pred_mean)@np.diag(weights_c[:,0])@(x_pred_set-x_pred_mean).T
    return x_pred_mean,Px_

def mle_x(y,u,f,H,x_prev,w,gamma2,Px_prev,Pw):
    dimx = x0.shape[0]
    C = covariance_x_w(x_prev,Px_prev,w,Pw,f,u)
    
    if C.shape[0]>1:
        invC = np.linalg.inv(C)
    else:
        invC = 1/C
        
    def negllx(var):
        x = var.reshape((dimx,1))
        mux = x-f(w,x_prev,u)
        ll =(y-H@x)**2/gamma2+mux.T@invC@mux
        return ll
    
    res = basinhopping(func = negllx,x0=x_prev.flatten())
    
    x_= res.x.reshape((dimx,1))
    Px_ = np.linalg.inv(H.T@H/gamma2+invC)
    return x_,Px_

def predict_x(x_prev,Px_prev,u,f,w,Pw):
    x_ = f(w,x_prev,u)
    Px_ = covariance_x_w(x_prev,Px_prev,w,Pw,f,u)
    return x_,Px_

def propogate_x(w,x_init,f,us,C):
    dim_w = w.shape[0]
    def pred(wx,u):
        w = wx[:dim_w,:]
        x_ = wx[dim_w:,:]
        x_next = f(w,x_,u)
        if np.isnan(x_next).any():
            return np.vstack([w,x_])
        return np.vstack([w,x_next])
    
    wx = np.vstack([w,x_init])
    for i in range(us.shape[1]):
        wx,C = covariance_x(wx,C,pred,us[:,[i]])
    return wx,C

def conditional_mle(model,ys,us,w0,x_init0,invC0,Gamma0,lr,n_iter,nonneg_x = False):
    block_size = ys.shape[1]
    usn = torch.tensor(us)
    ysn = torch.tensor(ys)
    
    wx_prev = torch.tensor(np.vstack([w0,x_init0]))
    invC = torch.tensor(invC0)
    
    if Gamma0.shape[0]==1:
        invGamma2 = torch.tensor(1/Gamma0)
    else:
        invGamma2 = torch.linalg.inv(torch.tensor(Gamma0))
    
    
    def negll(x_init_,w_):
        if nonneg_x:
            x_ = x_init_**2
        else:
            x_ = x_init_
            
        #prior
        mux = torch.vstack([w_,x_]) - wx_prev
        ll = 0.5*mux.T@invC@mux
        
        for i in range(usn.shape[1]):
            muy = ysn[:,[i]]-model.predict(x_)
            ll += (0.5*muy.T@invGamma2@muy)
            x_ = model.transition(w_,x_,usn[:,[i]])
        return ll
    
    def loss(wx):
        w_ = wx[:w0.shape[0],:]
        x_ = wx[w0.shape[0]:,:]
        
        mux = torch.vstack([w_,x_]) - wx_prev
        ll = 0.5*mux.T@invC@mux
        
        for i in range(usn.shape[1]):
            muy = ysn[:,[i]]-model.predict(x_)
            ll += (0.5*muy.T@invGamma2@muy)
            x_ = model.transition(w_,x_,usn[:,[i]])
            
        return ll
    
    w = torch.tensor(w0,requires_grad=True)
    if nonneg_x:
        x_init = torch.tensor(np.sqrt(np.maximum(np.zeros(x_init0.shape),x_init0)),requires_grad=True)
    else:
        x_init = torch.tensor(x_init0,requires_grad=True)
    optw = AdamW([w,x_init],lr = lr)
    
    lossesw = []  
    min_loss = negll(x_init,w).clone().detach()
    bestw = w.clone().detach()
    bestx = x_init.clone().detach()
    
    
    for i in range(n_iter):
        optw.zero_grad()
        lossw = negll(x_init,w)
        lossw.backward()
        optw.step()
        
        with torch.no_grad():
            if lossw<min_loss:
                min_loss = lossw
                bestw = w.clone().detach()
                bestx = x_init.clone().detach()
                
            lossesw.append(lossw.item())

    if nonneg_x:
        x_init1 =  bestx.detach().numpy()**2
    else:
        x_init1 =  bestx.detach().numpy()
    
    w1 = bestw.detach().numpy()
    
    
    
    #FIM1 = torch.autograd.functional.hessian(loss,torch.tensor(np.vstack([w1,x_init1]))).squeeze(-1).squeeze(1).detach().numpy()
    #FIM1 = (FIM1+FIM1.T)/2
    
    FIM2 = model.fim(torch.tensor(w1),torch.tensor(x_init1),usn.clone().detach())[0].detach().numpy()+invC0
   
    
    #print('F1',FIM1,np.linalg.pinv(FIM1))

    #print('F2',FIM2,np.linalg.pinv(FIM2))
        
    return x_init1,w1,FIM2,lossesw

def recursive_mle(model,ys,us,block_size,w0,x_init0,invC0,Gamma0,lr,n_iter,nonneg_x=False):
    N = ys.shape[1]
    print('N=',N)
    if N%block_size==0:
        x_init1,w1,invC1,losses = conditional_mle(model,ys[:,-block_size:],us[:,-block_size:],w0,x_init0,invC0,Gamma0,lr,n_iter,nonneg_x)
         
            
        print('guess',x_init0,'est',x_init1,'y_init',ys[:,[-block_size]])
        print('update',us[:,-block_size:],'from',us)
       
        #print('before propogate C',invC1,scipy.linalg.pinv(invC1))
        #wx,C = propogate_analys(torch.tensor(w1),torch.tensor(x_init1),model.transition,model.transition_jac,torch.tensor(us[:,-block_size:]),torch.tensor(scipy.linalg.pinv(invC1)))
        wx,C = propogate_x(w1,x_init1,model.transition_np,us[:,-block_size:],scipy.linalg.pinv(invC1))
        #print('after propogate C',C)
        w1 = wx[:w1.shape[0],:]
        x1 = wx[w1.shape[0]:,:]
        x_init_prop = x1.copy()
        invC1 = scipy.linalg.pinv(C)
    else:
        return x_init0,x_init0,w0,invC0
    
    print('loss',losses[0],losses[n_iter//2],losses[-2],losses[-1])
    return x_init_prop,x_init1,w1,invC1

def recursive_mle_opt(model,ys,us,block_size,w0,x_init0,invC0,lr,n_iter):
    N = ys.shape[1]
    print('N=',N)
    if N%block_size==0:
        def func(wx,wx_prev,invC_prev,invGamma,ys,us,H,H2,f,dim_theta):
                wx=wx.reshape(-1,1)
                mux = wx-wx_prev
                ll = 0.5*mux.T@invC_prev@mux
                w_ = wx[:dim_theta,:]
                x_ = wx[dim_theta:,:]

                for i in range(us.shape[1]):

                    muy = ys[:,[i]] - H@x_

                    ll+= 0.5*muy.T@invGamma@muy
                    x_ = f(w_,x_,us[:,[i]])
                return ll
            
            
        dim_theta = w0.shape[0]
        wx = np.vstack([w0,x_init0])
        minimizer_kwargs = {"method": "L-BFGS-B","args":(wx.copy(),invC0,model.invGamma_np,ys[:,-block_size:],us[:,-block_size:],model.H2,model.H3,model.transition_np,dim_theta)}
        ret = basinhopping(func, wx.copy(), minimizer_kwargs=minimizer_kwargs,niter=n_iter,stepsize=0.2)
        
        if ret.lowest_optimization_result.success == False and ret.lowest_optimization_result.nit<10:
            print('fail redo',ret.x,ret.fun)
            minimizer_kwargs = {"method": "L-BFGS-B","args":(wx.copy(),invC0,model.invGamma_np,ys[:,-block_size:],us[:,-block_size:],model.H2,model.H3,model.transition_np,dim_theta)}
            ret = basinhopping(func, wx.copy(), minimizer_kwargs=minimizer_kwargs,niter=5*n_iter,T=5,stepsize=0.1)
        
        print(ret)
        wx_ = ret.x
        w1 = wx_[:dim_theta].reshape(-1,1)
        x_init1 = wx_[dim_theta:].reshape(-1,1)
        invC1 = model.fim(torch.tensor(w1),torch.tensor(x_init1),torch.tensor(us[:,-block_size:]))[0].detach().numpy()+invC0
        print('guess',x_init0,'est',x_init1,'y_init',ys[:,[-block_size]])
    
        #print('before propogate C',invC1,scipy.linalg.pinv(invC1))
        #wx,C = propogate_analys(torch.tensor(w1),torch.tensor(x_init1),model.transition,model.transition_jac,torch.tensor(us[:,-block_size:]),torch.tensor(scipy.linalg.pinv(invC1)))
        wx,C = propogate_x(w1,x_init1,model.transition_np,us[:,-block_size:],scipy.linalg.pinv(invC1))
        #print('after propogate C',C)
        w1 = wx[:w1.shape[0],:]
        x1 = wx[w1.shape[0]:,:]
        x_init_prop = x1.copy()
        invC1 = scipy.linalg.pinv(C)
    else:
        return x_init0,x_init0,w0,invC0
    
    return x_init_prop,x_init1,w1,invC1


def recursive_mle_sigma(model,ys,us,block_size,w0,x_init0,invC0,gamma0,invS0,n_iter):
    #Gamma suppose to be a vector
    
    N = ys.shape[1]
    if N%block_size==0:
        zs = np.zeros((ys.shape[0],block_size))
        x_ = x_init0
        for i in range(block_size):
            zs[:,[i]] = ys[:,[-block_size+i]] - model.H2@x_
            x_ = model.transition_np(w0,x_,us[:,[-block_size+i]])
        
        
        def func(gamma,gamma_prev,invS_prev,zs):
                gamma2 = (gamma**2)
                mu_sig = (gamma2.reshape(-1,1)-gamma_prev)
                ll = 0.5*mu_sig.T@invS_prev@mu_sig + 0.5*block_size*np.sum(np.log(gamma2))
                
                invGamma = np.diag(1/gamma2)
         
                for i in range(zs.shape[1]):
                    ll+= 0.5*zs[:,[i]].T@invGamma@zs[:,[i]]
                    
                return ll
            
        minimizer_kwargs = {"method": "L-BFGS-B","args":(gamma0,invS0,zs)}
        ret = basinhopping(func, gamma0.flatten(), minimizer_kwargs=minimizer_kwargs,niter=n_iter)
        
        if ret.lowest_optimization_result.success == False and ret.lowest_optimization_result.nit<10:
            print('fail redo')
            minimizer_kwargs = {"method": "L-BFGS-B","args":(gamma0,invS0,zs)}
            ret = basinhopping(func, gamma0, minimizer_kwargs=minimizer_kwargs,niter=2*n_iter,stepsize=0.3)
        
       
        
        gamma1 = (ret.x**2).reshape(-1,1)
        invS1 = invS0 + 0.5*block_size*np.diag(1/(gamma1**2))
    else:
        return gamma0,invS0
    
    return gamma1,invS1  

def ekf_opt(model,y,u,w0,x0,invC0):
            
    dim_theta = w0.shape[0]
    P0 = scipy.linalg.pinv(nearestPD(invC0))
    F = model.transition_jac_np(w0,x0,u)
    H = model.H3
    R = np.linalg.inv(model.invGamma_np)
    
    
    #prediction-step
    x_pred = model.transition_np(w0,x0,u)
    wx0 = np.vstack([w0,x_pred])
    P_pred = F@P0@F.T #+Q --> no process noise
    
    
    
    #innovation-step
    y_pred = model.predict_np(x_pred)
    S = H@P_pred@H.T + R
    W = P_pred@H.T@np.linalg.inv(S)
    
    #update
    wx1 = wx0 + W@(y-y_pred)
    P1 = P_pred - W@S@W.T
    invC1= scipy.linalg.pinv(nearestPD(P1))
    
    w1 = wx1[:dim_theta,:]
    x1 = wx1[dim_theta:,:]
    
    return x1,w1,invC1

def ekf_prop(model,u,w,x0,invC0):
    x1 = model.transition_np(w,x0,u)
    F = model.transition_jac_np(w,x0,u)
    P = np.linalg.pinv(invC0)
    P_prop = F@P@F.T
    invC1 = np.linalg.pinv(P_prop)
    return x1,invC1

class ModelFixedWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module,theta:Tensor, x0:Tensor,fim_prev:Tensor,C_prev:Tensor):
        super().__init__()
        self.add_module("model", model)
        self.register_buffer("theta", theta)  
        self.register_buffer("x0", x0)
        self.register_buffer("fim_prev", fim_prev)
        self.register_buffer("C_prev", C_prev)

    def forward(self, uk:Tensor)-> Tensor:
        return self.model.forward(uk, self.theta, self.x0, self.fim_prev,self.C_prev)
    

import time
def generate_uk_ss(model,estimator,T,K,us,theta0,x_init0,invC0,gamma0,invS0,simulator,Q,block_size,weight='full',dt=1,seed=None,n_iter = 20000,lr_desc=8e-4):
    #warm-start
    us_ = us.copy()
    _,ys = simulator(us_,Q,seed=seed,dt=dt)
    if len(ys.shape)==1:
        ys = ys.reshape(1,-1)
    us_ = us_.reshape(1,-1)
    N0 = us_.shape[1]
    
    ysn = torch.tensor(np.array(ys))
    usn = torch.tensor(np.array(us_))
    
    theta = theta0.copy()
    M = theta.shape[0]
    x_init = x_init0.copy()
    invC = invC0.copy()
    uk0 = model.init_uk(torch.tensor(theta),torch.tensor(x_init),K)
    theta_delta0 = 0.01*np.random.randn(theta.shape[0]).reshape(theta.shape)
    gamma = gamma0.copy()
    invS = invS0.copy()
    #update model gamma
    model.gamma2 = gamma
    model.invGamma = torch.diag(1/torch.tensor(gamma).flatten())
    model.invGamma_np = np.diag(1/gamma.flatten())
    
    t = 1
    for i in tnrange(T-N0):
        N = usn.shape[1]
        #estimate
        if N%block_size==0:
            print('update theta')
            x_init,theta,invC,gamma,invS = estimator(model,block_size,x_init,theta,invC,gamma,invS,ys,us_,t)
            
            #update model gamma
            model.gamma2 = gamma
            model.invGamma = torch.diag(1/torch.tensor(gamma).flatten())
            model.invGamma_np = np.diag(1/gamma.flatten())
            t+=1
            
        
        theta_torch = torch.tensor(theta)
        x_init_torch = torch.tensor(x_init)
        invC_torch = torch.tensor(invC)
        C_torch = torch.tensor(scipy.linalg.pinv(invC))
        
        if weight=='diag':
            W = torch.diag(torch.diag(C_torch))
        else:
            W = C_torch
            
        if N%block_size>0:
            # i.e. T_min = 4 N = 9 block_size = 2--> x_init == x[:,[4+2+2]]-> us[[4+2+2]:[4+2+2+1]]
            us_remains = usn[:,-(N%block_size):]
            #model.cache(theta_torch,x_init_torch,us_remains)
            #print('cache',us_remains,'from',usn)
            
            wx,C_prop = propogate_x(theta,x_init,model.transition_np,us_remains.numpy(),np.linalg.pinv(invC))
            #wx,C_prop = propogate_analys(torch.tensor(theta),torch.tensor(x_init),model.transition,model.transition_jac,us_remains,torch.tensor(np.linalg.pinv(invC)))
            print('propogate N=',N,wx[theta.shape[0]:,:])#,C_prop,xw2[theta.shape[0]:,:],C_prop2)
            theta_torch = torch.tensor(wx[:theta.shape[0],:])
            x_init_torch = torch.tensor(wx[theta.shape[0]:,:])
            invC_torch = torch.linalg.pinv(torch.tensor(C_prop))
            
            
            if weight=='diag':
                W = torch.diag(torch.diag(torch.tensor(C_prop)))
            else:
                W = torch.tensor(C_prop)
                
        
        t1 = time.time()
        criterion_fixed = torch.jit.script(ModelFixedWrapper(model, theta_torch, x_init_torch,invC_torch,W))
        t2 = time.time()

        print(f"JIT time: {(t2 - t1):.3f} s")
        uk,losses = minimize(criterion_fixed,uk0,[],lr=lr_desc,n_iter=n_iter)   
        #uk,losses = minimize([model.forward],uk0,[theta_torch,x_init_torch,invC_torch,W],lr=lr_desc,n_iter=n_iter)   
        
        t3 = time.time()
        print(f"Design time: {(t3 - t2):.3f} s")
        
        print('control',uk,len(losses),losses[0],losses[len(losses)//2],losses[-2],losses[-1])
            
        #get first u as control unit, reuse remains as u0
        us_= np.hstack([us_,model.repara_u(uk.detach()).numpy()[:,[0]]])
        usn = torch.tensor(us_)
        uk0[:,:-1] = uk.clone().detach().numpy()[:,1:]
            
        try:
            x_,ys = simulator(us_.flatten(),Q,seed=seed,dt=dt)
        except:
            try:
                x_,ys = simulator(us_.flatten(),Q,seed=seed,dt=dt)
            except:
                print('?',us_.flatten())
        if len(ys.shape)==1:
            ys = ys.reshape(1,-1)
        ysn = torch.tensor(ys)
            
    return us_.flatten()

def generate_uk_ss_opt(model,estimator,T,K,us,theta0,x_init0,invC0,gamma0,invS0,simulator,Q,block_size,weight='full',dt=1,seed=None,n_iter = 200):
    #warm-start
    us_ = us.copy()
    _,ys = simulator(us_,Q,seed=seed,dt=dt)
    if len(ys.shape)==1:
        ys = ys.reshape(1,-1)
    us_ = us_.reshape(1,-1)
    N0 = us_.shape[1]
    
    
    theta = theta0.copy()
    M = theta.shape[0]
    x_init = x_init0.copy()
    gamma = gamma0.copy()
    invS = invS0.copy()
    invC = invC0.copy()
    uk0 = model.init_uk(torch.tensor(theta),torch.tensor(x_init),K) #TODO 
    bounds = [model.u_range for i in range(K)]
    model.gamma2 = gamma
    model.invGamma = torch.diag(1./torch.tensor(model.gamma2.flatten()))
    model.invGamma_np = np.diag(1./model.gamma2.flatten())
    
    t = 1
    for i in tnrange(T-N0):
        N = us_.shape[1]
        #estimate
        if N%block_size==0:
            print('update theta/gamma')
            x_init,theta,invC,gamma,invS = estimator(model,block_size,x_init,theta,invC,gamma,invS,ys,us_,t)
            model.gamma2 = gamma
            model.invGamma = torch.diag(1./torch.tensor(model.gamma2.flatten()))
            model.invGamma_np = np.diag(1./model.gamma2.flatten())
            #print(model.gamma2,model.invGamma.shape,model.invGamma_np.shape)
            t+=1
            
        
        theta_prop = theta.copy()
        x_init_prop = x_init.copy()
        fim_prop = invC.copy()
        
        if weight=='diag':
            W = np.diag(np.diag(scipy.linalg.pinv(invC)))
        else:
            W = scipy.linalg.pinv(invC).copy()
            
        
        if N%block_size>0:
            us_remains = us_[:,-(N%block_size):]

            wx,C_prop = propogate_x(theta,x_init,model.transition_np,us_remains,np.linalg.pinv(invC))
            fim_prop = scipy.linalg.pinv(C_prop).copy()
            
            if weight=='diag':
                W = np.diag(np.diag(C_prop))
            else:
                W = C_prop
                
            print('propogate N=',N,wx[theta.shape[0]:,:])
            
            theta_prop = wx[:theta.shape[0],:].copy()
            x_init_prop = wx[theta.shape[0]:,:].copy()
   

        def func(uk,theta,x0,fim_prev,W):
            return model.control_criterion_np(uk.reshape(1,-1),theta,x0,fim_prev,W)
        
        #uk0 = model.best_binary_guess(K,theta_prop,x_init_prop,fim_prop,W)
        #print('best binary guess',uk0)
        
        print('control loss before',func(uk0,theta_prop,x_init_prop,fim_prop,W),'eigval',model.eig_fim_np(uk0,theta_prop,x_init_prop,fim_prop))
        minimizer_kwargs = {"method": "L-BFGS-B","bounds":bounds,"args":(theta_prop,x_init_prop,fim_prop,W)}
        ret = basinhopping(func,uk0, minimizer_kwargs=minimizer_kwargs,niter=200)
        uk = ret.x.reshape(1,-1) 
        print('control loss after',uk,func(uk,theta_prop,x_init_prop,fim_prop,W),'eigval',model.eig_fim_np(uk,theta_prop,x_init_prop,fim_prop))
            
        #get first u as control unit, reuse remains as u0
        us_= np.hstack([us_,uk[:,[0]]])
        
        #uk0[:,:-1] = uk[:,1:]
        #print('copy uk0',uk0) 
            
        try:
            x_,ys = simulator(us_.flatten(),Q,seed=seed,dt=dt)
        except:
            try:
                x_,ys = simulator(us_.flatten(),Q,seed=seed,dt=dt)
            except:
                print('?',us_.flatten())
        if len(ys.shape)==1:
            ys = ys.reshape(1,-1)
        
            
    return us_.flatten()


def generate_uk_ss_ekf(model,estimator,T,K,us,theta0,x_init0,invC0,gamma0,invS0,simulator,Q,block_size,weight='full',dt=1,seed=None,n_iter = 200):
    #warm-start
    us_ = us.copy()
    _,ys = simulator(us_,Q,seed=seed,dt=dt)
    if len(ys.shape)==1:
        ys = ys.reshape(1,-1)
    us_ = us_.reshape(1,-1)
    N0 = us_.shape[1]
    
    
    theta = theta0.copy()
    M = theta.shape[0]
    x_init = x_init0.copy()
    gamma = gamma0.copy()
    invS = invS0.copy()
    invC = invC0.copy()
    uk0 = model.init_uk(torch.tensor(theta),torch.tensor(x_init),K) #TODO 
    bounds = [model.u_range for i in range(K)]
    model.gamma2 = gamma
    model.invGamma = torch.diag(1./torch.tensor(model.gamma2.flatten()))
    model.invGamma_np = np.diag(1./model.gamma2.flatten())
    
    t = 1
    for i in tnrange(1,T):
        #estimate
        print('i',i,'y',ys[:,i])
        x_init,theta,invC,gamma,invS = estimator(model,block_size,x_init,theta,invC,gamma,invS,ys[:,:i+1],us_[:,:i+1],t)
        model.gamma2 = gamma
        model.invGamma = torch.diag(1./torch.tensor(model.gamma2.flatten()))
        model.invGamma_np = np.diag(1./model.gamma2.flatten())
        t+=1
            
        if i<N0-1:
            continue
            
        theta_prop = theta.copy()
        wx,C_prop = propogate_x(theta,x_init,model.transition_np,us_[:,[i]],scipy.linalg.pinv(invC))
        fim_prop = scipy.linalg.pinv(C_prop).copy()
        #x_init_prop,fim_prop = ekf_prop(model,us_[:,[i]],theta,x_init,invC)
        x_init_prop = wx[theta.shape[0]:,:].copy()
        print('i',i+1,'prop',x_init_prop)
        
        if weight=='diag':
            W = np.diag(np.diag(C_prop))
        else:
            W = C_prop.copy()
            
   

        def func(uk,theta,x0,fim_prev,W):
            return model.control_criterion_np(uk.reshape(1,-1),theta,x0,fim_prev,W)
        
        print('control loss before',func(uk0,theta_prop,x_init_prop,fim_prop,W),'eigval',model.eig_fim_np(uk0,theta_prop,x_init_prop,fim_prop))
        minimizer_kwargs = {"method": "L-BFGS-B","bounds":bounds,"args":(theta_prop,x_init_prop,fim_prop,W)}
        ret = basinhopping(func,uk0, minimizer_kwargs=minimizer_kwargs,niter=200)
        uk = ret.x.reshape(1,-1) 
        print('control loss after',uk,func(uk,theta_prop,x_init_prop,fim_prop,W),'eigval',model.eig_fim_np(uk,theta_prop,x_init_prop,fim_prop))
            
        #get first u as control unit, reuse remains as u0
        us_= np.hstack([us_,uk[:,[0]]])
        
        uk0[:,:-1] = uk[:,1:]
            
        try:
            x_,ys = simulator(us_.flatten(),Q,seed=seed,dt=dt)
        except:
            try:
                x_,ys = simulator(us_.flatten(),Q,seed=seed,dt=dt)
            except:
                print('?',us_.flatten())
        if len(ys.shape)==1:
            ys = ys.reshape(1,-1)
        
            
    return us_.flatten()