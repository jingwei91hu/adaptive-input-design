import numpy as np
import torch

from torch import Tensor
from typing import Any,Tuple


@torch.jit.script
def fimJ_tank_fused(
    theta: Tensor,
    x0: Tensor,
    uk: Tensor,
    H: Tensor,
    invGamma: Tensor,
    dt: float,
    A1: float,
    A2: float,
    x12: Tensor,
    xw13: Tensor,
    xw21: Tensor,
) -> Tuple[Tensor, Tensor]:
    N = uk.shape[1]
    M1 = theta.shape[0]
    M2 = x0.shape[0]

    dx_theta = torch.zeros((M2, M1), dtype=theta.dtype, device=theta.device)
    dx_x0 = torch.eye(M2, dtype=theta.dtype, device=theta.device)
    x = x0
    I_tensor = torch.empty((N, M1 + M2, M1 + M2), dtype=theta.dtype, device=theta.device)
    yk = torch.empty((M2,N), dtype=x0.dtype, device=x0.device)
    for i in range(N):
        # Compute d_mu
        d_mu = H @ torch.hstack([dx_theta, dx_x0])

        # Transition
        x_ = torch.relu(x)
        sqrt_x1 = torch.sqrt(torch.clamp(x_[0, 0], min=1e-8))
        sqrt_x2 = torch.sqrt(torch.clamp(x_[1, 0], min=1e-8))

        q12 = theta[1, 0] * sqrt_x1
        x1_next = x_[0, 0] + (theta[0, 0] * uk[0, i] - q12) * dt / A1
        x2_next = x_[1, 0] + (q12 - theta[2, 0] * sqrt_x2) * dt / A2
        x = torch.relu(torch.stack([x1_next, x2_next]).unsqueeze(1))
        yk[:,[i]] =  H@x
        
        # Mask zero outputs
        d_mu[0, :] *= (x[0, 0] != 0)
        d_mu[1, :] *= (x[1, 0] != 0)

        I_tensor[i] = d_mu.T @ invGamma @ d_mu

        # Compute Jacobians
        x11 = 1 - 0.5 * dt * theta[1, 0] / sqrt_x1 / A1
        x21 = 0.5 * dt * theta[1, 0] / sqrt_x1 / A2
        x22 = 1 - 0.5 * dt * theta[2, 0] / sqrt_x2 / A2

        dxx = torch.stack([
            torch.stack([x11, x12]),
            torch.stack([x21, x22])
        ])

        xw11 = uk[0, i] * dt / A1
        xw12 = -sqrt_x1 * dt / A1
        xw22 = sqrt_x1 * dt / A2
        xw23 = -sqrt_x2 * dt / A2

        dxtheta = torch.stack([
            torch.stack([xw11, xw12, xw13]),
            torch.stack([xw21, xw22, xw23])
        ])

        dx_theta = dxx @ dx_theta + dxtheta
        dx_x0 = dxx @ dx_x0

    return I_tensor.sum(dim=0),yk

@torch.jit.script
def control_criterion_tank_fused(
    uk: Tensor,
    theta: Tensor,
    x0: Tensor,
    fim_prev: Tensor,
    C_prev: Tensor,
    H: Tensor,
    invGamma: Tensor,
    dt: float,
    A1: float,
    A2: float,
    x12: Tensor,
    xw13: Tensor,
    xw21: Tensor,
) -> Tensor:
    I_,yk = fimJ_tank_fused(theta, x0, uk, H, invGamma, dt, A1, A2, x12, xw13, xw21)
    I = I_ + fim_prev
    M = theta.shape[0]
    A = I[:M, :M]
    B = I[:M, M:]
    D = I[M:, M:]
    C_theta = C_prev[:M, :M]

    schur = A - B @ torch.linalg.solve(D, B.T)
    trace_val = torch.trace(torch.linalg.solve(schur, C_theta))
    return trace_val


@torch.jit.script
def barrier_term(yk:Tensor,y_range:Tensor,y_norm:Tensor,yfactor:float):
    y_dim,K = yk.shape
    l = torch.nn.functional.relu(y_range[:,0,:]-yk)
    u = torch.nn.functional.relu(yk-y_range[:,1,:])
    return ((yfactor*(l+u)/y_norm)**2).sum()/K


@torch.jit.script
def control_criterion_tank_fused_barrier(
    uk: Tensor,
    theta: Tensor,
    x0: Tensor,
    fim_prev: Tensor,
    C_prev: Tensor,
    H: Tensor,
    invGamma: Tensor,
    dt: float,
    A1: float,
    A2: float,
    x12: Tensor,
    xw13: Tensor,
    xw21: Tensor,
    y_range:Tensor
    ,y_norm:Tensor
    ,yfactor:float
) -> Tensor:
    I_,yk = fimJ_tank_fused(theta, x0, uk, H, invGamma, dt, A1, A2, x12, xw13, xw21)
    I = I_ + fim_prev
    M = theta.shape[0]
    A = I[:M, :M]
    B = I[:M, M:]
    D = I[M:, M:]
    C_theta = C_prev[:M, :M]

    schur = A - B @ torch.linalg.solve(D, B.T)
    trace_val = torch.trace(torch.linalg.solve(schur, C_theta))
    return trace_val+barrier_term(yk,y_range,y_norm,yfactor)



class StateSpaceTwoTankModelJ(torch.nn.Module):
    
    def __init__(self,u_range,gamma2,dt):
        super().__init__()
        self.register_buffer("u_range", torch.tensor(u_range))
        self.register_buffer("invGamma", torch.diag(torch.tensor(1/gamma2.flatten())))
        self.dt = dt
        self.A1 = 0.1
        self.A2 = 0.1
        self.register_buffer("H", torch.eye(2))
        self.register_buffer("CI", torch.eye(5))
        self.register_buffer("eps", torch.zeros((2,1)))
        self.register_buffer("x12",torch.tensor(0.))
        self.register_buffer("xw13",torch.tensor(0.))
        self.register_buffer("xw21",torch.tensor(0.))
    
    @torch.jit.unused
    def init_numpy(self):
        self.H2 = np.eye(2)
        self.H3 = np.array([[0.,0.,0.,1.,0.],[0.,0.,0.,0.,1.]])
        
    @torch.jit.unused
    def init_uk(self,theta,x_prev,K):
        return 0.03*np.random.randn(K).reshape((1,K))
    
    @torch.jit.unused
    def predict(self,x):
        return self.H@x

    @torch.jit.unused   
    def transition(self,theta,x,u):
        x_ = torch.maximum(x,torch.zeros((2,1)))
        
        q12 = theta[1,:]*torch.sqrt(x_[0,:]+1.0e-64)
        x1_ = x_[0,:] + (theta[0]*u[0,:]- q12)*self.dt/self.A1
        x2_ = x_[1,:] + (q12-theta[2,:]*torch.sqrt(x_[1,:]+1.0e-64))*self.dt/self.A2
        
        return torch.maximum(torch.vstack([x1_,x2_]),torch.zeros((2,1)))
    
    @torch.jit.unused   
    def transition_np(self,theta,x,u):
        x_ = np.maximum(x,np.zeros((2,1)))
        
        q12 = theta[1,:]*np.sqrt(x_[0,:]+1.0e-64)
        x1_ = x_[0,:] + (theta[0]*u[0,:]- q12)*self.dt/self.A1
        x2_ = x_[1,:] + (q12-theta[2,:]*np.sqrt(x_[1,:]+1.0e-64))*self.dt/self.A2
        
        return np.maximum(np.vstack([x1_,x2_]),np.zeros((2,1)))
    
    
    @torch.jit.unused
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
    
    @torch.jit.unused
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
    
    @torch.jit.unused
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
            yk.append(self.H@x_)
            
        return torch.concatenate(I).sum(axis=0),x_,dx_theta,dx_x0,torch.hstack(yk)

    def repara_u(self,u:Tensor)->Tensor:
        return torch.sigmoid(u).mul(self.u_range[1]-self.u_range[0])+self.u_range[0]
    
   # @torch.jit.export
    def forward(self,uk:Tensor,  theta:Tensor, x0:Tensor, fim_prev:Tensor,C_prev:Tensor) -> Tensor:
    #     #return self.control_criterion(self.repara_u(uk),theta, x0, fim_prev,C_prev)
    #     return forward_fused(
    #     uk, theta, x0, fim_prev, C_prev,
    #     self.H,
    #     self.invGamma,
    #     self.dt,
    #     self.A1,
    #     self.A2,
    #     self.u_range,
    #     self.x12,
    #     self.xw13,
    #     self.xw21
    # )
        # Reparameterize control input u
        uk_real = torch.sigmoid(uk) * (self.u_range[1] - self.u_range[0]) +self.u_range[0]

        return control_criterion_tank_fused(
            uk_real,
            theta,
            x0,
            fim_prev,
            C_prev,
            self.H,
            self.invGamma,
            self.dt,
            self.A1,
            self.A2,
            self.x12,
            self.xw13,
            self.xw21
        )


    
class BarrierSSTwoTankModelJ(torch.nn.Module):
    
    def __init__(self,u_range,gamma2,dt,y_range,K,y_fac):
        super().__init__()
        self.register_buffer("u_range", torch.tensor(u_range))
        self.register_buffer("invGamma", torch.diag(torch.tensor(1/gamma2.flatten())))
        self.dt = dt
        self.A1 = 0.1
        self.A2 = 0.1
        self.yfactor = y_fac
        self.register_buffer("H", torch.eye(2))
        self.register_buffer("CI", torch.eye(5))
        self.register_buffer("eps", torch.zeros((2,1)))
        self.register_buffer("x12",torch.tensor(0.))
        self.register_buffer("xw13",torch.tensor(0.))
        self.register_buffer("xw21",torch.tensor(0.))
        self.register_buffer("y_range",torch.tensor(y_range).unsqueeze(2).repeat(1, 1, K))
        self.register_buffer("y_norm",torch.tensor((y_range[:, 1] - y_range[:, 0])).unsqueeze(1).repeat(1, K))
       
        
    @torch.jit.unused
    def init_numpy(self):
        self.H2 = np.eye(2)
        self.H3 = np.array([[0.,0.,0.,1.,0.],[0.,0.,0.,0.,1.]])
        self.y_range_np = self.y_range.numpy()
        self.y_norm_np =  self.y_norm.numpy()
        
    @torch.jit.unused
    def init_uk(self,theta,x_prev,K):
        return 0.03*np.random.randn(K).reshape((1,K))
    
    @torch.jit.unused
    def predict(self,x):
        return self.H@x

    @torch.jit.unused   
    def transition(self,theta,x,u):
        x_ = torch.maximum(x,torch.zeros((2,1)))
        
        q12 = theta[1,:]*torch.sqrt(x_[0,:]+1.0e-64)
        x1_ = x_[0,:] + (theta[0]*u[0,:]- q12)*self.dt/self.A1
        x2_ = x_[1,:] + (q12-theta[2,:]*torch.sqrt(x_[1,:]+1.0e-64))*self.dt/self.A2
        
        return torch.maximum(torch.vstack([x1_,x2_]),torch.zeros((2,1)))
    
    @torch.jit.unused   
    def transition_np(self,theta,x,u):
        x_ = np.maximum(x,np.zeros((2,1)))
        
        q12 = theta[1,:]*np.sqrt(x_[0,:]+1.0e-64)
        x1_ = x_[0,:] + (theta[0]*u[0,:]- q12)*self.dt/self.A1
        x2_ = x_[1,:] + (q12-theta[2,:]*np.sqrt(x_[1,:]+1.0e-64))*self.dt/self.A2
        return np.maximum(np.vstack([x1_,x2_]),np.zeros((2,1)))
    
    @torch.jit.unused
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
    
    @torch.jit.unused
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
    
    @torch.jit.unused
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
            yk.append(self.H@x_)
            
        return torch.concatenate(I).sum(axis=0),x_,dx_theta,dx_x0,torch.hstack(yk)

    def repara_u(self,u:Tensor)->Tensor:
        return torch.sigmoid(u).mul(self.u_range[1]-self.u_range[0])+self.u_range[0]
    
   # @torch.jit.export
    def forward(self,uk:Tensor,  theta:Tensor, x0:Tensor, fim_prev:Tensor,C_prev:Tensor) -> Tensor:
            # Reparameterize control input u
        uk_real = torch.sigmoid(uk) * (self.u_range[1] - self.u_range[0]) + self.u_range[0]

        return control_criterion_tank_fused_barrier(
            uk_real,
            theta,
            x0,
            fim_prev,
            C_prev,
            self.H,
            self.invGamma,
            self.dt,
            self.A1,
            self.A2,
            self.x12,
            self.xw13,
            self.xw21,
            self.y_range,
            self.y_norm,
            self.yfactor
        )

########################Pendulum##############

@torch.jit.script
def fimJ_pendulum_fused(
    theta: Tensor,
    x0: Tensor,
    uk: Tensor,
    H: Tensor,
    invGamma: Tensor,
    dt: float,
    xx1: Tensor
) -> Tuple[Tensor, Tensor]:
    N = uk.shape[1]
    M1 = theta.shape[0]
    M2 = x0.shape[0]
    M3 = H.shape[0]

    dx_theta = torch.zeros((M2, M1))
    dx_x0 = torch.eye(M2)
    x = x0.clone()

    I_tensor = torch.zeros((M1 + M2, M1 + M2))
    yk = torch.empty((M3, N))

    for i in range(N):
        d_mu = H @ torch.cat((dx_theta, dx_x0), dim=1)
        I_tensor += d_mu.T @ invGamma @ d_mu

        u_i = uk[:, i]              # (1,)
        sin_x = torch.sin(x[0, 0])  # scalar Tensor
        cos_x = torch.cos(x[0, 0])

        # ∂f/∂x
        x21 = theta[0, 0] * cos_x * dt
        row2_dxx = torch.stack([x21, torch.tensor(1.0)])
        dxx = torch.stack([xx1, row2_dxx])

        # ∂f/∂theta
        xw21 = sin_x * dt
        xw22 = u_i[0] * dt
        row2_dxtheta = torch.stack([xw21, xw22])
        row1_dxtheta = torch.zeros_like(row2_dxtheta)
        dxtheta = torch.stack([row1_dxtheta, row2_dxtheta])  # shape (2,2)

        dx_theta = dxx @ dx_theta + dxtheta
        dx_x0 = dxx @ dx_x0

        # Transition
        dx1 = x[1, :] * dt              # shape: (1,)
        dx2 = (theta[0, :] * sin_x + theta[1, :] * u_i) * dt  # shape: (1,)
        dx = torch.cat([dx1, dx2], dim=0).unsqueeze(1)        # shape: (2,1)
        x = x + dx

        # Prediction
        yk[:, i] = (H @ x).reshape(-1)

    return I_tensor, yk


class StateSpacePendulumModelJ(torch.nn.Module):
    
    def __init__(self,u_range,gamma2,dt):
        super().__init__()
        self.register_buffer("u_range", torch.tensor(u_range))
        self.register_buffer("invGamma", torch.diag(torch.tensor(1/gamma2.flatten())))
        self.dt = dt
        self.register_buffer("H", torch.tensor([[1.,0.]]))
        self.register_buffer("CI", torch.eye(4))
        self.register_buffer("xx1", torch.tensor([1.,self.dt]))
        
    @torch.jit.unused
    def init_numpy(self):
        self.H2 = np.array([[1.,0.]])
        self.H3 = np.array([[0,0.,1.,0.]])
        self.CI2 = np.eye(4)     
        
    @torch.jit.unused
    def init_uk(self,theta,x_prev,K):
        return 0.03*np.random.randn(K).reshape((1,K))
    
    @torch.jit.unused
    def predict(self,x):
        return self.H@x
    
    @torch.jit.unused
    def transition(self,theta,x,u):
        dx1 = x[1,:]*self.dt
        dx2 = (theta[0,:]*torch.sin(x[0,:]) + theta[1,:]*u[0,:])*self.dt 
        dx = torch.vstack([dx1,dx2])
        return x+dx
    
    @torch.jit.unused
    def transition_np(self,theta,x,u):
        dx1 = x[1,:]*self.dt
        dx2 = (theta[0,:]*np.sin(x[0,:]) + theta[1,:]*u[0,:])*self.dt
        dx = np.vstack([dx1,dx2])
        return x+dx
    
    @torch.jit.unused
    def dxx(self,theta,x,u): 
        x21 = theta[0,0]*torch.cos(x[0,0])*self.dt
        x22 = torch.tensor(1)
        return torch.vstack([self.xx1,torch.hstack([x21,x22])])
    
   
    @torch.jit.unused
    def dxtheta(self,theta,x,u): 
        xw21 = torch.sin(x[0,0])*self.dt
        xw22 = u[0,0]*self.dt
        return torch.vstack([torch.zeros(2),torch.hstack([xw21,xw22])])
    
    @torch.jit.unused
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
                    
            I.append((d_mu.T@self.invGamma@d_mu).unsqueeze(0))
            
            xx = self.dxx(theta,x_,uk[:,[i]])
            xw = self.dxtheta(theta,x_,uk[:,[i]])
            dx_theta = xx@dx_theta + xw
            dx_x0 = xx@dx_x0
            x_ = x_pred.clone()
            yk.append(self.H@x_)
            
        return torch.concatenate(I).sum(axis=0),x_,dx_theta,dx_x0,torch.hstack(yk)

    def repara_u(self,u:Tensor)->Tensor:
        return torch.sigmoid(u).mul(self.u_range[1]-self.u_range[0])+self.u_range[0]
    
   
    def forward(self,uk:Tensor,  theta:Tensor, x0:Tensor, fim_prev:Tensor,C_prev:Tensor) -> Tensor:
        # Reparameterize control input u
        uk_real = torch.sigmoid(uk) * (self.u_range[1] - self.u_range[0]) +self.u_range[0]
        I_,yk = fimJ_pendulum_fused(theta, x0, uk_real, self.H, self.invGamma, self.dt, self.xx1)
        I = I_ + fim_prev
        M = theta.shape[0]
        A = I[:M, :M]
        B = I[:M, M:]
        D = I[M:, M:]
        C_theta = C_prev[:M, :M]

        schur = A - B @ torch.linalg.solve(D, B.T)
        trace_val = torch.trace(torch.linalg.solve(schur, C_theta))
        return trace_val

    





class BarrierPendulumModelJ(torch.nn.Module):
    
    def __init__(self,u_range,gamma2,dt,y_range,K,y_fac):
        super().__init__()
        self.register_buffer("u_range", torch.tensor(u_range))
        self.register_buffer("invGamma", torch.diag(torch.tensor(1/gamma2.flatten())))
        self.dt = dt
        self.yfactor = y_fac
        self.register_buffer("H", torch.tensor([[1.,0.]]))
        self.register_buffer("CI", torch.eye(4))
        self.register_buffer("xx1", torch.tensor([1.,self.dt]))
        self.register_buffer("y_range",torch.tensor(y_range).unsqueeze(2).repeat(1, 1, K))
        self.register_buffer("y_norm",torch.tensor((y_range[:, 1] - y_range[:, 0])).unsqueeze(1).repeat(1, K))
        
    @torch.jit.unused
    def init_numpy(self):
        self.H2 = np.array([[1.,0.]])
        self.H3 = np.array([[0,0.,1.,0.]])
        self.CI2 = np.eye(4)     
        self.y_range_np = self.y_range.numpy()
        self.y_norm_np =  self.y_norm.numpy()
        
    @torch.jit.unused
    def init_uk(self,theta,x_prev,K):
        return 0.03*np.random.randn(K).reshape((1,K))
    
    @torch.jit.unused
    def predict(self,x):
        return self.H@x
    
    @torch.jit.unused
    def transition(self,theta,x,u):
        dx1 = x[1,:]*self.dt
        dx2 = (theta[0,:]*torch.sin(x[0,:]) + theta[1,:]*u[0,:])*self.dt 
        dx = torch.vstack([dx1,dx2])
        return x+dx
    
    @torch.jit.unused
    def transition_np(self,theta,x,u):
        dx1 = x[1,:]*self.dt
        dx2 = (theta[0,:]*np.sin(x[0,:]) + theta[1,:]*u[0,:])*self.dt
        dx = np.vstack([dx1,dx2])
        return x+dx
    
    @torch.jit.unused
    def dxx(self,theta,x,u): 
        x21 = theta[0,0]*torch.cos(x[0,0])*self.dt
        x22 = torch.tensor(1)
        return torch.vstack([self.xx1,torch.hstack([x21,x22])])
    
   
    @torch.jit.unused
    def dxtheta(self,theta,x,u): 
        xw21 = torch.sin(x[0,0])*self.dt
        xw22 = u[0,0]*self.dt
        return torch.vstack([torch.zeros(2),torch.hstack([xw21,xw22])])
    
    @torch.jit.unused
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
                    
            I.append((d_mu.T@self.invGamma@d_mu).unsqueeze(0))
            
            xx = self.dxx(theta,x_,uk[:,[i]])
            xw = self.dxtheta(theta,x_,uk[:,[i]])
            dx_theta = xx@dx_theta + xw
            dx_x0 = xx@dx_x0
            x_ = x_pred.clone()
            yk.append(self.H@x_)
            
        return torch.concatenate(I).sum(axis=0),x_,dx_theta,dx_x0,torch.hstack(yk)

    def repara_u(self,u:Tensor)->Tensor:
        return torch.sigmoid(u).mul(self.u_range[1]-self.u_range[0])+self.u_range[0]
    
   
     
   # @torch.jit.export
    def forward(self,uk:Tensor,  theta:Tensor, x0:Tensor, fim_prev:Tensor,C_prev:Tensor) -> Tensor:
        # Reparameterize control input u
        uk_real = torch.sigmoid(uk) * (self.u_range[1] - self.u_range[0]) + self.u_range[0]    
        I_,yk = fimJ_pendulum_fused(theta, x0, uk_real, self.H, self.invGamma, self.dt, self.xx1)
        I = I_ + fim_prev
        M = theta.shape[0]
        A = I[:M, :M]
        B = I[:M, M:]
        D = I[M:, M:]
        C_theta = C_prev[:M, :M]

        schur = A - B @ torch.linalg.solve(D, B.T)
        trace_val = torch.trace(torch.linalg.solve(schur, C_theta))
        return trace_val+barrier_term(yk,self.y_range,self.y_norm,self.yfactor)

