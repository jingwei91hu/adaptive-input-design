import numpy as np
import torch
import itertools
from abc import ABC,abstractmethod


class ScalarGaussianModel:
    @abstractmethod
    def predict(self,theta,ys,us,**kparams):
        pass
    
    @abstractmethod
    def estimate_criterion(self,theta,ys,us,**kparams):
        pass
    
    @abstractmethod
    def control_criterion(self,theta,ys,us,**kparams):
        pass
    
    @abstractmethod
    def d_mu(self,theta,ys,us,**kparams):
        pass
    
    def neg_log_likelihood(self,mu,gamma2):
        return (mu@mu.T)/gamma2 + mu.shape[0]*torch.log(gamma2)
    
    def neg_log_likelihood_batch(self,mu,gamma2):
        return (mu/gamma2@mu.T)/mu.shape[0] + torch.log(gamma2).mean()
    

def repara_power_spectrum(var,c,freq_weights):
    N = len(var)
    dft = torch.fft.fft(var)
    dft_ = dft*c/torch.clamp(torch.real(dft@freq_weights@torch.conj(dft))/N,min=c)
    return torch.real(torch.fft.ifft(dft_))

def repara_frequency(var,c):
    dft = torch.fft.fft(var)

    dft_ = dft*c/torch.clamp(torch.real(dft*torch.conj(dft)),min=c)
    
    return torch.real(torch.fft.ifft(dft_))
     
def repara_ellipsoid(var,invC,q):
    return var*q/torch.clamp(var.T@invC@var,min=q)

def repara_sigmoid(var,lower,upper):
    return torch.sigmoid(var).mul(upper-lower)+lower

def roll(x,shifts):
    N = x.shape[0]
    if x.shape[0]<=shifts:
        return torch.zeros(N)
    return torch.concat([torch.zeros(shifts),x[:N-shifts]])


class StateSpaceLinearModel:
    
    def __init__(self,u_range,gamma2):
        self.u_range = u_range
        self.gamma2 = gamma2
        self.invGamma = torch.ones((1,1))/gamma2
        self.invGamma_np = np.ones((1,1))/gamma2
        
        self.H = torch.tensor([[1.,0.]])
        self.H2 = np.array([[1.,0.]])
        self.H3 = np.array([[0.,0.,1.,0]])
        self.e1 = torch.tensor([[1.],[0]])
        self.A_ = torch.tensor([[0,1.],[-0.1,0.5]]) 
        
        self.e3 = torch.tensor([[0,1.],[0.,0.]])
        
        self.e4 = torch.tensor([[1.],[0]])
        self.e5 = torch.tensor([[0.,1]])
        self.CI = torch.eye(4)
        self.CI2 = np.eye(4)
        
        
    
    def transition(self,theta,x,u):
        A = self.A_ + (theta*self.e1)@self.e1.T
        B = self.e3@theta
        return A@x+B@u
    
    def transition_jac(self,theta,x,u):
        M1 = theta.shape[0]
        M2 = x.shape[0]
        F = torch.eye(M1+M2)      
        F[M1:,:M1] = self.dxtheta(theta,x,u)
        F[M1:,M1:] = self.dxx(theta,x,u)
        return F
    
    def transition_jac_np(self,theta,x,u):
        M1 = theta.shape[0]
        M2 = x.shape[0]
        F = np.eye(M1+M2)      
        F[M1:,:M1] = self.dxtheta_np(theta,x,u)
        F[M1:,M1:] = self.dxx_np(theta,x,u)
        return F
    
    def transition_np(self,theta,x,u):
        A = self.A_.numpy() + (theta*self.e1.numpy())@self.e1.numpy().T
        B = self.e3.numpy()@theta
        return A@x+B@u
    
    def predict(self,x):
        return self.H@x
    
    def predict_np(self,x):
        return self.H2@x
    
    def predict_series(self,theta,x0,uk):
        x_ = x0.clone()
        yk = []
        for i in range(uk.shape[1]):
            yk.append(self.predict(x_))
            x_ = self.transition(theta,x_,uk[:,[i]])
        return torch.hstack(yk)
    
    def dxx(self,theta,x,u):
        return self.A_ + (theta*self.e1)@self.e1.T
    
    def dxx_np(self,theta,x,u):
        pass
    
    def dxtheta(self,theta,x,u):
        return (x*self.e1)@self.e1.T+self.e4@u@self.e5
    
    def dxtheta_np(self,theta,x,u):
        pass
    
    def dmu(self,theta,x0,uk,x_=None,dx_theta=None,dx_x0=None):
        N = uk.shape[1]
        M1 = theta.shape[0]
        M2 = x0.shape[0]
        M3 = self.H.shape[0]
        d_mu = torch.zeros((N,M3,M1+M2))
        
        if dx_theta is None:
            dx_theta = torch.zeros((M2,M1))
            
        if dx_x0 is None:
            dx_x0 = torch.eye(M2)
            
        if x_ is None:
            x_ = x0.clone()
            
        for i in range(N):
            d_mu[i,:,:M1] = self.H@dx_theta
            d_mu[i,:,M1:] = self.H@dx_x0
            
            xx = self.dxx(theta,x_,uk[:,[i]])
            xw = self.dxtheta(theta,x_,uk[:,[i]])
            dx_theta = xx@dx_theta + xw
            dx_x0 = xx@dx_x0
            x_ = self.transition(theta,x_,uk[:,[i]])
        return d_mu,x_,dx_theta,dx_x0

    def fim(self, theta, x0, uk,x_=None,dx_theta=None,dx_x0=None):
        N = uk.shape[1]
        M1 = theta.shape[0]
        M2 = x0.shape[0]
        M3 = self.H.shape[0]
        
        if N==0:
            return torch.zeros((M1+M2,M1+M2))
        
        if dx_theta is None:
            dx_theta = torch.zeros((M2,M1))
            
        if dx_x0 is None:
            dx_x0 = torch.eye(M2)
            
        if x_ is None:
            x_ = x0.clone()
    
        I = []
        yk = []
        for i in range(N):
            d_mu = self.H@torch.hstack([dx_theta,dx_x0])
            I.append((d_mu.T@self.invGamma@d_mu).unsqueeze(0))
            
            xx = self.dxx(theta,x_,uk[:,[i]])
            xw = self.dxtheta(theta,x_,uk[:,[i]])
            dx_theta = xx@dx_theta + xw
            dx_x0 = xx@dx_x0
            x_ = self.transition(theta,x_,uk[:,[i]])
            yk.append(self.predict(x_))
            
        return torch.concatenate(I).sum(axis=0),x_,dx_theta,dx_x0,torch.hstack(yk)
    
    def fim_np(self, theta, x0, uk):
        N = uk.shape[1]
        M1 = theta.shape[0]
        M2 = x0.shape[0]
        M3 = self.H2.shape[0]
        
        
        dx_theta = np.zeros((M2,M1))
        dx_x0 = np.eye(M2)
        x_ = x0.copy()
    
        I = np.zeros((M1+M2,M1+M2))
        yk = []
        for i in range(N):
            d_mu = self.H2@np.hstack([dx_theta,dx_x0])
            I+=(d_mu.T@self.invGamma_np@d_mu)
            
            xx = self.dxx_np(theta,x_,uk[:,[i]])
            xw = self.dxtheta_np(theta,x_,uk[:,[i]])
            dx_theta = xx@dx_theta + xw
            dx_x0 = xx@dx_x0
            x_ = self.transition_np(theta,x_,uk[:,[i]])
            yk.append(self.H2@x_)
            
        return I,np.hstack(yk)
    
    def cache(self,theta,x0,us):
        if us is not None and us.shape[1]>0:
            I,x_,dx_theta,dx_x0,_ = self.fim(theta,x0,us)
            self.fim_prev = I.clone().detach()+ torch.eye(theta.shape[0]+x0.shape[0])*1e-8
            self.x_ = x_.clone().detach()
            self.dx_theta = dx_theta.clone().detach()
            self.dx_x0 = dx_x0.clone().detach()
        else:
            self.fim_prev = None
            self.x_ = None
            self.dx_theta = None
            self.dx_x0 = None
            
    def control_criterion_np(self, uk, theta, x0,fim_prev,C_prev):
        I_,yk = self.fim_np(theta,x0,uk)
        I = I_+fim_prev
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 =  (A-B@np.linalg.inv(D)@C)@C_prev[:M,:M]
        return np.trace(np.linalg.inv(I2))
    
    
    def eig_fim_np(self,uk,theta,x0,fim_prev):
        I_,_ = self.fim_np(theta,x0,uk)
        I = I_+fim_prev
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        return np.linalg.eigvals(A-B@np.linalg.inv(D)@C)
    
    def eig_fim(self,uk,theta,x0,fim_prev):
        I_,_,_,_,yk = self.fim(theta,x0,uk)
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        return torch.linalg.eigvals(A-B@np.linalg.inv(D)@C)
    
    def control_criterion(self, uk, theta, x0,fim_prev,C_prev):
        I_,_,_,_,yk = self.fim(theta,x0,uk)
        I = I_+fim_prev
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 =  (A-B@torch.linalg.inv(D)@C)@C_prev[:M,:M]
        return torch.trace(torch.linalg.inv(I2))
    
    def control_step(self,uk,theta_delta,  theta, x0, fim_prev,C_prev):    
        uk_ = self.repara_u(uk)
        return self.control_criterion(uk_,theta, x0, fim_prev,C_prev)
    
    def repara_u(self,u):
        return repara_sigmoid(u,*self.u_range)
    
    def recompute_u_range(self,theta,x_prev,K):
        return self.u_range
    
    def init_uk(self,theta,x_prev,K):
        return 0.03*np.random.randn(K).reshape((1,K))
    
    def best_binary_guess(self,K,theta,x0,fim_prev,C_prev):
        best_cost = np.inf
        best_u_cand = None
        for u_cand in itertools.product(*[self.u_range for _ in range(K)]):
            cost = self.control_criterion_np(np.array([u_cand]),theta,x0,fim_prev,C_prev)
            if cost<best_cost:
                best_u_cand = np.array([u_cand])
                best_cost = cost
        return best_u_cand
    
    def compute_likelihood(self,theta,x_init,ys,us):
        x_ = x_init.clone()
        l = 0
        for i in range(us.shape[1]):
            muy = ys[:,[i]] - self.predict(x_)
            l+=(-0.5)*(muy.T@muy)/gamma2
            x_ = self.transition(theta,x_,us[:,[i]])
        return l
    
    # def hessian_of_criterion(self,theta,x_init,ys,us):
    #     def cl(v):
    #         w = v[:theta.shape[0],:]
    #         x = v[theta.shape[0]:,:]
    #         return self.compute_likelihood(w,x,ys,us)
    #     hess = torch.autograd.functional.hessian(cl,torch.vstack([theta,x_init]))
    #     return -hess.squeeze(1).squeeze(-1)

class BarrierModel:
    def __init__(self,y_range,K,M,yfac=5):
        self.y_range_np = np.tile(np.expand_dims(y_range,2),(1,1,K))
        self.y_range = torch.tensor(self.y_range_np)
        self.yfac = yfac
        self.K = K
        self.M = M

    def barrier_term(self,yk):
        y_dim,K = yk.shape
        sc = self.M/K

        l = torch.nn.functional.relu(self.y_range[:,0,:]-yk)
        u = torch.nn.functional.relu(yk-self.y_range[:,1,:])
        b = torch.exp(((self.yfac*(l+u))**2).sum(axis=0)).sum()

        return b*sc    

    def barrier_term_np(self,yk):
        y_dim,K = yk.shape
        sc = self.M/K
        l =np.maximum(np.zeros((y_dim,K)),self.y_range_np[:,0,:]-yk)
        u =np.maximum(np.zeros((y_dim,K)),yk-self.y_range_np[:,1,:])
        b=np.exp(((self.yfac*(l+u))**2).sum(axis=0)).sum()
        return b*sc       
    
class BarrierModel2:
    def __init__(self,y_range,K,yfac=5):
        self.y_range_np = np.tile(np.expand_dims(y_range,2),(1,1,K))
        self.y_range = torch.tensor(self.y_range_np)
        self.y_norm_np =  np.tile(np.expand_dims(y_range[:,1]-y_range[:,0],axis=1),(1,K))
        self.y_norm = torch.tensor(self.y_norm_np)
        self.yfac = yfac
        self.K = K

    def barrier_term(self,yk):
        y_dim,K = yk.shape
        sc = 1/K

        l = torch.nn.functional.relu(self.y_range[:,0,:]-yk)
        u = torch.nn.functional.relu(yk-self.y_range[:,1,:])
        b=((self.yfac*(l+u)/self.y_norm)**2).sum()
        return b*sc    

    def barrier_term_np(self,yk):
        y_dim,K = yk.shape
        sc = 1/K
        
        l =np.maximum(np.zeros((y_dim,K)),self.y_range_np[:,0,:]-yk)
        u =np.maximum(np.zeros((y_dim,K)),yk-self.y_range_np[:,1,:])
        b=((self.yfac*(l+u)/self.y_norm_np)**2).sum()
        return b*sc 
    
class BarrierStateSpaceLinearModel(StateSpaceLinearModel,BarrierModel):
    
    def __init__(self,u_range,gamma2,y_range,K,M):
        super(BarrierStateSpaceLinearModel,self).__init__(u_range=u_range,gamma2=gamma2,y_range=y_range,K=K,M=M)
        
    
    def control_criterion(self, uk, theta, x0,fim_prev,C_prev):
        return super().control_criterion(uk, theta, x0,fim_prev,C_prev)+self.barrier_term(yk)
    
    def control_criterion_np(self, uk, theta, x0,fim_prev,C_prev):
        return super().control_criterion_np(uk, theta, x0,fim_prev,C_prev)+self.barrier_term_np(yk)