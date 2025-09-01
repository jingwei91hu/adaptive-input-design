import numpy as np
import torch
import scipy

from .simu import *
from .model_base import *


class StateSpaceCruiseControlModel(StateSpaceLinearModel):
    #state x velocity
    def __init__(self,u_range,gamma2,dt):
        self.u_range = u_range
        self.gamma2 = gamma2
        self.invGamma = torch.ones((1,1))/gamma2
        self.invGamma_np = np.ones((1,1))/gamma2
        self.dt = dt 
        self.H = torch.tensor([[1.]])
        self.H2 = np.array([[1.]])
        self.H3 = np.array([[0.,0.,1.]])
        self.CI = torch.eye(3)
        self.CI2 = np.eye(3)
        
    # x_{t+1} = x_{t} + u*dt + kx_{t}^2*dt
    def transition(self,theta,x,u):
        return x + theta[0,0]*u*self.dt+theta[1,0]*x*torch.abs(x)*self.dt
    
    def transition_np(self,theta,x,u):
        return x + theta[0,0]*u*self.dt+theta[1,0]*x*np.abs(x)*self.dt
    
    def dxx(self,theta,x,u):
         #x*sign(x) + |x|
        return 1 + self.dt*theta[1,0]*(2*torch.abs(x))
    
    def dxx_np(self,theta,x,u):
         #x*sign(x) + |x|
        return 1 + self.dt*theta[1,0]*(2*np.abs(x))
    
    def dxtheta(self,theta,x,u):
        xw1 = u*self.dt
        xw2 = x*torch.abs(x)*self.dt
        return torch.hstack([xw1,xw2])
    
    def dxtheta_np(self,theta,x,u):
        xw1 = u*self.dt
        xw2 = x*np.abs(x)*self.dt
        return np.hstack([xw1,xw2])
    
#old
class StateSpaceCruiseControlModel2(StateSpaceCruiseControlModel):
    def control_criterion(self, uk, theta, x0,fim_prev,C_prev):
        I_,_,_,_,yk = self.fim(theta,x0,uk)
        I = I_@C_prev+self.CI
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 =  (A-B@torch.linalg.inv(D)@C)
        return torch.trace(torch.linalg.inv(I2))
    
    def control_criterion_np(self, uk, theta, x0,fim_prev,C_prev):
        I_ = self.fim_np(theta,x0,uk)[0]
        I = I_@C_prev+self.CI2
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 =  (A-B@np.linalg.pinv(D)@C)
        return np.trace(np.linalg.pinv(I2))

#D-optimal
class StateSpaceCruiseControlModel3(StateSpaceCruiseControlModel):
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
    
    
class BarrierSSCruiseControlModel(StateSpaceCruiseControlModel,BarrierModel):
    
    def __init__(self,u_range,gamma2,dt,y_range,K,M):
        StateSpaceCruiseControlModel.__init__(self, u_range,gamma2,dt)
        BarrierModel.__init__(self,y_range,K,M)
        
    
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
    
    
class BarrierSSCruiseControlModel2(StateSpaceCruiseControlModel,BarrierModel2):
    
    def __init__(self,u_range,gamma2,dt,y_range,K,yfac):
        StateSpaceCruiseControlModel.__init__(self, u_range,gamma2,dt)
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
    
#A-opt
class BarrierSSCruiseControlModel3(StateSpaceCruiseControlModel,BarrierModel2):
    
    def __init__(self,u_range,gamma2,dt,y_range,K,yfac):
        StateSpaceCruiseControlModel.__init__(self, u_range,gamma2,dt)
        BarrierModel2.__init__(self,y_range,K,yfac)
        
    def control_criterion(self, uk, theta, x0,fim_prev,C_prev):
        I_,_,_,_,yk = self.fim(theta,x0,uk)
        I = I_+fim_prev
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 =  (A-B@torch.linalg.inv(D)@C)
        return torch.trace(torch.linalg.inv(I2)) + self.barrier_term(yk)
    
    def control_criterion_np(self, uk, theta, x0,fim_prev,C_prev):
        I_,yk = self.fim_np(theta,x0,uk)
        I = I_+fim_prev
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        I2 =  (A-B@np.linalg.inv(D)@C)
        return np.trace(np.linalg.inv(I2))+ self.barrier_term_np(yk)