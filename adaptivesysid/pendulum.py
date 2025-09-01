import numpy as np
import torch
import scipy

from .simu import *
from .model_base import *

class StateSpacePendulumModel(StateSpaceLinearModel):
    def __init__(self,u_range,gamma2,dt):
        self.gamma2 = gamma2
        self.invGamma = torch.ones((1,1))/gamma2
        self.invGamma_np = np.ones((1,1))/gamma2
        self.u_range = u_range
        self.dt = dt 
        self.H = torch.tensor([[1.,0.]])
        self.H2 = np.array([[1.,0.]])
        self.H3 = np.array([[0,0.,1.,0.]])
        self.CI = torch.eye(4)
        self.CI2 = np.eye(4)
        self.xx1 = torch.tensor([1.,self.dt])
        
    def transition(self,theta,x,u):
        dx1 = x[1,:]*self.dt
        dx2 = (theta[0,:]*torch.sin(x[0,:]) + theta[1,:]*u[0,:])*self.dt 
        dx = torch.vstack([dx1,dx2])
        return x+dx
    
    def transition_np(self,theta,x,u):
        dx1 = x[1,:]*self.dt
        dx2 = (theta[0,:]*np.sin(x[0,:]) + theta[1,:]*u[0,:])*self.dt
        dx = np.vstack([dx1,dx2])
        return x+dx
    
    def dxx(self,theta,x,u): 
        x21 = theta[0,0]*torch.cos(x[0,0])*self.dt
        x22 = torch.tensor(1)
        return torch.vstack([self.xx1,torch.hstack([x21,x22])])
    
    def dxx_np(self,theta,x,u):
        return np.array([[1.,self.dt],[theta[0,0]*np.cos(x[0,0])*self.dt,1.]])
        
    def dxtheta(self,theta,x,u): 
        xw21 = torch.sin(x[0,0])*self.dt
        xw22 = u[0,0]*self.dt
        return torch.vstack([torch.zeros(2),torch.hstack([xw21,xw22])])
    
    def dxtheta_np(self,theta,x,u):
        return np.array([[0.,0.],[np.sin(x[0,0])*self.dt,u[0,0]*self.dt]])
    
    def control_criterion_np(self, uk, theta, x0,fim_prev,C_prev):
        I_ = self.fim_np(theta,x0,uk)[0]
        I =  I_ + fim_prev #I_@C_prev + self.CI
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 = A-B@np.linalg.inv(D)@C
        
        J_inv= np.linalg.inv(I2@C_prev[:M,:M])
        return np.trace(J_inv)
    
    def control_criterion(self, uk, theta, x0,fim_prev,C_prev):
        I_ = self.fim(theta,x0,uk)[0]
        I =  I_ + fim_prev #I_@C_prev + self.CI
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 = A-B@torch.linalg.inv(D)@C
        J_inv= torch.linalg.inv(I2@C_prev[:M,:M])
        return torch.trace(J_inv)
    
#A-optimal    
class StateSpacePendulumModel2(StateSpacePendulumModel):
    def control_criterion_np(self, uk, theta, x0,fim_prev,C_prev):
        I_ = self.fim_np(theta,x0,uk)[0]
        I =  I_ + fim_prev #I_@C_prev + self.CI2
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 = A-B@np.linalg.inv(D)@C
        return np.trace(np.linalg.inv(I2))
    
    def control_criterion(self, uk, theta, x0,fim_prev,C_prev):
        I_ = self.fim(theta,x0,uk)[0]
        I =  I_ + fim_prev #I_@C_prev + self.CI
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 = A-B@torch.linalg.inv(D)@C
        return torch.trace(torch.linalg.inv(I2))
    
#old
class StateSpacePendulumModel3(StateSpacePendulumModel):
    
    def control_criterion_np(self, uk, theta, x0,fim_prev,C_prev):
        I_ = self.fim_np(theta,x0,uk)[0]
        I =  I_@C_prev + self.CI2
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 = A-B@np.linalg.inv(D)@C
        return np.trace(np.linalg.inv(I2))
    
    def control_criterion(self, uk, theta, x0,fim_prev,C_prev):
        I_ = self.fim(theta,x0,uk)[0]
        I =  I_@C_prev + self.CI
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 = A-B@torch.linalg.inv(D)@C
        return torch.trace(torch.linalg.inv(I2))
    
#D-optimal    
class StateSpacePendulumModel4(StateSpacePendulumModel):
    def control_criterion(self, uk, theta, x0,fim_prev,C_prev):
        I_,_,_,_,yk = self.fim(theta,x0,uk)
        I = I_+fim_prev
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 =  (A-B@torch.linalg.inv(D)@C)
        return -torch.logdet(I2)
    
    def control_criterion_np(self, uk, theta, x0,fim_prev,C_prev):
        I_ = self.fim_np(theta,x0,uk)[0]
        I = I_+fim_prev
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 =  (A-B@np.linalg.pinv(D)@C)
        sign, logabsdet = np.linalg.slogdet(I2)
        return -sign*logabsdet

class BarrierPendulumModel(StateSpacePendulumModel,BarrierModel):
    
    def __init__(self,u_range,gamma2,dt,y_range,K,M,yfac):
        StateSpacePendulumModel.__init__(self, u_range,gamma2,dt)
        BarrierModel.__init__(self,y_range,K,M,yfac)
        
    def control_criterion(self, uk, theta, x0,fim_prev,C_prev):
        I_,_,_,_,yk = self.fim(theta,x0,uk)
        I = I_+fim_prev
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 =  (A-B@torch.linalg.inv(D)@C)@C_prev[:M,:M]
        return torch.trace(torch.linalg.inv(I2)) + self.barrier_term(yk)
    
    def control_criterion_np(self, uk, theta, x0,fim_prev,C_prev):
        I_,yk = self.fim_np(theta,x0,uk)
        I = I_+fim_prev
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 =  (A-B@np.linalg.inv(D)@C)@C_prev[:M,:M]
        return np.trace(np.linalg.inv(I2))+self.barrier_term_np(yk)
    
    
class BarrierPendulumModel2(StateSpacePendulumModel,BarrierModel2):
    
    def __init__(self,u_range,gamma2,dt,y_range,K,yfac):
        StateSpacePendulumModel.__init__(self, u_range,gamma2,dt)
        BarrierModel2.__init__(self,y_range,K,yfac)
        
    def control_criterion(self, uk, theta, x0,fim_prev,C_prev):
        I_,_,_,_,yk = self.fim(theta,x0,uk)
        I = I_+fim_prev
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 =  (A-B@torch.linalg.inv(D)@C)@C_prev[:M,:M]
        return torch.trace(torch.linalg.inv(I2)) + self.barrier_term(yk)
    
    def control_criterion_np(self, uk, theta, x0,fim_prev,C_prev):
        I_,yk = self.fim_np(theta,x0,uk)
        I = I_+fim_prev
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 =  (A-B@np.linalg.inv(D)@C)@C_prev[:M,:M]
        return np.trace(np.linalg.inv(I2))+ self.barrier_term_np(yk)