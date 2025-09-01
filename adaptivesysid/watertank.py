import numpy as np
import torch
import scipy

from .simu import *
from .model_base import *

class StateSpaceTwoTankModel(StateSpaceLinearModel):
    #state x velocity
    def __init__(self,u_range,gamma2,dt):
        self.u_range = u_range
        self.gamma2 = gamma2
        self.invGamma = torch.diag(torch.tensor(1/gamma2.flatten()))
        self.invGamma_np = np.diag(1/gamma2.flatten())
        self.dt = dt 
        self.H = torch.eye(2)
        self.A1 = 0.1
        self.A2 = 0.1
        self.CI = torch.eye(5)
        self.H2 = np.eye(2)
        self.H3 = np.array([[0.,0.,0.,1.,0.],[0.,0.,0.,0.,1.]])
        self.CI2 = np.eye(5)
        
        self.eps =torch.zeros((2,1)) 
        
        self.x12 = torch.tensor(0.)
        self.xw13 = torch.tensor(0.)
        self.xw21 = torch.tensor(0.)
        
    def transition(self,theta,x,u):
    
        x_ = torch.maximum(x,torch.zeros((2,1)))
        
        q12 = theta[1,:]*torch.sqrt(x_[0,:]+1.0e-64)
        x1_ = x_[0,:] + (theta[0]*u[0,:]- q12)*self.dt/self.A1
        x2_ = x_[1,:] + (q12-theta[2,:]*torch.sqrt(x_[1,:]+1.0e-64))*self.dt/self.A2
        
        return torch.maximum(torch.vstack([x1_,x2_]),torch.zeros((2,1)))
    
    def transition_np(self,theta,x,u):
        x_ = np.maximum(x,np.zeros((2,1)))
        
        q12 = theta[1,:]*np.sqrt(x_[0,:]+1.0e-64)
        x1_ = x_[0,:] + (theta[0]*u[0,:]- q12)*self.dt/self.A1
        x2_ = x_[1,:] + (q12-theta[2,:]*np.sqrt(x_[1,:]+1.0e-64))*self.dt/self.A2
        
        return np.maximum(np.vstack([x1_,x2_]),np.zeros((2,1)))
    
    def dxx(self,theta,x,u):
        if x[0,0]>0:
            sqrtx1 = torch.sqrt(x[0,0])
            x11 = 1-0.5*self.dt*theta[1,0]/sqrtx1/self.A1
            x21 = 0.5*self.dt*theta[1,0]/sqrtx1/self.A2
        else:
            x11 = torch.tensor(0.)
            x21 = torch.tensor(0.)
        
        if x[1,0]>0:
            x22 = 1-0.5*self.dt*theta[2,0]/torch.sqrt(x[1,0])/self.A2   
        else:
            x22 = torch.tensor(0.)
        
        ret= torch.vstack([torch.hstack([x11,self.x12]),torch.hstack([x21,x22])])
        
        return ret
    
    def dxx_np(self,theta,x,u):
        dXX = np.zeros((x.shape[0],x.shape[0]))
        
        if x[0,0]>0:
            sqrtx1 = np.sqrt(x[0,0])
            dXX[0,0] = 1-0.5*self.dt*theta[1,0]/sqrtx1/self.A1
            dXX[1,0] = 0.5*self.dt*theta[1,0]/sqrtx1/self.A2

        if x[1,0]>0:
            dXX[1,1] = 1-0.5*self.dt*theta[2,0]/np.sqrt(x[1,0])/self.A2   
        return dXX
    
    def dxtheta(self,theta,x,u):
        xw11 = u[0,0]*self.dt/self.A1
            
        
        if x[1,0]>0:
            xw23 = -torch.sqrt(x[1,0])*self.dt/self.A2
        else:
            xw23 = torch.tensor(0.)
        
        if x[0,0]>0:
            xw12 = -torch.sqrt(x[0,0])*self.dt/self.A1
            xw22 = torch.sqrt(x[0,0])*self.dt/self.A2
        else:
            xw12 = torch.tensor(0.)
            xw22 = torch.tensor(0.)
            
        
        ret= torch.vstack([torch.hstack([xw11,xw12,self.xw13]),torch.hstack([self.xw21,xw22,xw23])])
        return ret
    
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
            
            x_pred = self.transition(theta,x_,uk[:,[i]])
            
            if x_pred[0,0]==0:
                d_mu[0,:] = 0
            if x_pred[1,0]==0:
                d_mu[1,:] = 0
                    
            I.append((d_mu.T@self.invGamma@d_mu).unsqueeze(0))
            
            xx = self.dxx(theta,x_,uk[:,[i]])
            xw = self.dxtheta(theta,x_,uk[:,[i]])
            dx_theta = xx@dx_theta + xw
            dx_x0 = xx@dx_x0
            x_ = x_pred.clone()
            yk.append(self.predict(x_))
            
        return torch.concatenate(I).sum(axis=0),x_,dx_theta,dx_x0,torch.hstack(yk)
    
    def dxtheta_np(self,theta,x,u):
        dXW = np.zeros((x.shape[0],theta.shape[0]))
        
        dXW[0,0] = u[0,0]*self.dt/self.A1
            
        
        if x[1,0]>0:
            dXW[1,2] = -np.sqrt(x[1,0])*self.dt/self.A2
        
        if x[0,0]>0:
            dXW[0,1] = -np.sqrt(x[0,0])*self.dt/self.A1
            dXW[1,1] = np.sqrt(x[0,0])*self.dt/self.A2
        return dXW
    
class StateSpaceTwoTankModel2(StateSpaceTwoTankModel):
    def control_criterion(self, uk, theta, x0,fim_prev,C_prev):
        I_,_,_,_,yk = self.fim(theta,x0,uk)
        I = I_@C_prev+self.CI
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        
        I2 = A-B@torch.linalg.inv(D)@C
        return torch.trace(torch.linalg.inv(I2))
    
    def control_criterion_np(self, uk, theta, x0, fim_prev,C_prev):
        I_ = self.fim_np(theta,x0,uk)[0]
        I = I_@C_prev+self.CI2
        M = theta.shape[0]
        A = I[:M,:M]
        B = I[:M,M:]
        C = I[M:,:M]
        D = I[M:,M:]
        
        I2 = A-B@scipy.linalg.pinv(D)@C
        return np.trace(scipy.linalg.pinv(I2))
#D-opt
class StateSpaceTwoTankModel3(StateSpaceTwoTankModel):
    
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

class BarrierSSTwoTankModel(StateSpaceTwoTankModel,BarrierModel):
    
    def __init__(self,u_range,gamma2,dt,y_range,K,M):
        StateSpaceTwoTankModel.__init__(self, u_range,gamma2,dt)
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
    
    
class BarrierSSTwoTankModel2(StateSpaceTwoTankModel,BarrierModel2):
    
    def __init__(self,u_range,gamma2,dt,y_range,K,yfac):
        StateSpaceTwoTankModel.__init__(self, u_range,gamma2,dt)
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