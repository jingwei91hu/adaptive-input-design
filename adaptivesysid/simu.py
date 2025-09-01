from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal,chi2
from scipy.special import expit
import scipy

import numpy as np
import control as ct
def white_noise(T,Q,dt=0,seed=None):
    """Generate a white noise signal with specified intensity.

    This function generates a (multi-variable) white noise signal of
    specified intensity as either a sampled continous time signal or a
    discrete time signal.  A white noise signal along a 1D array
    of linearly spaced set of times T can be computing using

        V = ct.white_noise(T, Q, dt)

    where Q is a positive definite matrix providing the noise intensity.

    In continuous time, the white noise signal is scaled such that the
    integral of the covariance over a sample period is Q, thus approximating
    a white noise signal.  In discrete time, the white noise signal has
    covariance Q at each point in time (without any scaling based on the
    sample time).

    """
    # Convert input arguments to arrays
    T = np.atleast_1d(T)
    Q = np.atleast_2d(Q)

    # Check the shape of the input arguments
    if len(T.shape) != 1:
        raise ValueError("Time vector T must be 1D")
    if len(Q.shape) != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Covariance matrix Q must be square")

    # Figure out the time increment
    #if dt != 0:
    #    # Discrete time system => white noise is not scaled
    #    dt = 1
    #else:
    #    dt = T[1] - T[0]

    # Make sure data points are equally spaced
    if not np.allclose(np.diff(T), T[1] - T[0]):
        raise ValueError("Time values must be equally spaced.")

    #np.random.seed(seed)
    # Generate independent white noise sources for each input
    #W = np.random.normal(0, 1/np.sqrt(dt), Q.shape[0]*1000).reshape((Q.shape[0],1000))

    # Return a linear combination of the noise sources
    #return sqrtm(Q) @ W[:,:T.size]
    ret =  np.random.default_rng(seed=seed).multivariate_normal(mean=np.zeros((Q.shape[0])), cov=Q, size=T.size)#multivariate_normal(np.zeros((Q.shape[0],)),Q,seed=seed,allow_singular=False).rvs(T.size)
    return ret.T

def simulate_nonlinear0(U,Q,x0=0,seed=None,dt=1):
    N = len(U)
    T = np.arange(1,N+1,dt)
    def poly(t,x,u,params):
        alpha = params.get('alpha',[1,-0.5])
        degree = len(alpha)
        ut = [alpha[i]*(u**(i+1)) for i in range(degree)]
        return sum(ut)

    pol = ct.NonlinearIOSystem(None,poly,inputs=1, outputs=1, dt=dt)
    sys_tf = ct.tf([0.,0.5],[1,-0.8],dt=dt)
    sys_hem = ct.series(pol,sys_tf)
    
    _,ys_,x = ct.input_output_response(sys_hem,T,U,x0,return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).flatten().copy()
    return x[0],ys

def simulate_nonlinear3(U,Q,x0=0,seed=None,dt=1):
    #zt = (u_{t-1}*0.6+u_t*0.8+0.3*y_{t-1})
    #yt = zt*(1-0.2*zt)
    N = len(U)
    T = np.arange(1,N+1,dt)
    def poly(t,x,u,params):
        alpha = params.get('alpha',[1,-0.2])
        degree = len(alpha)
        ut = [alpha[i]*(u**(i+1)) for i in range(degree)]
        return sum(ut)

    def time_delay_system(coef,delay, dt, inputs=1, outputs=1, **kwargs):
        """
        creates a pure time delay discrete-time system.
        time delay is equal to nearest whole number of `dt`s."""
        assert delay >= 0, "delay must be greater than or equal to zero"
        n = int(round(delay/dt))
        ninputs = inputs if isinstance(inputs, (int, float)) else len(inputs)
        assert ninputs == 1, "only one input supported"
        A = np.eye(n, k=-1)
        B = np.eye(n, 1)*coef
        C = np.eye(1, n, k=n-1)
        D = np.zeros((1,1))
        return ct.ss(A, B, C, D, dt, inputs=inputs, outputs=outputs, **kwargs)


    delayer1 = time_delay_system(0.3,dt,dt,inputs='y',outputs='w')
    delayer2 = time_delay_system(1,dt,dt,inputs='y',outputs='yd')
    delayers = ct.parallel([delayer1,delayer2])
    Us = ct.tf([0.8,0.6],[1,0],dt=dt,inputs='u',outputs='v',name='U')
    u_y = ct.summing_junction(inputs=['v','w'],output='d',name='S')
    F7 = ct.NonlinearIOSystem(None,poly, inputs='d', outputs='y', dt=dt,name='F')
    sys_wiener = ct.interconnect([Us,u_y,F7,delayer1,delayer2],inplist='u', outlist='yd',dt=dt)
    _,ys_,x = ct.input_output_response(sys_wiener,T,U,x0,return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).flatten().copy()
    return x[0],ys

def simulate_nonlinear5(U,Q,x0=0,seed=None,dt=1):
    #non-linear arx-sigmoid
    N = len(U)
    T = np.arange(1,N+1,dt)
    
    def sigmoid_stauration2(t,x,u,params):
        offset = params.get('Offset',4)
        scale = params.get('Scale',0.2)
        return scale*expit(u-offset)

    def time_delay_system(coef,delay, dt, inputs=1, outputs=1, **kwargs):
        """
        creates a pure time delay discrete-time system.
        time delay is equal to nearest whole number of `dt`s."""
        assert delay >= 0, "delay must be greater than or equal to zero"
        n = int(round(delay/dt))
        ninputs = inputs if isinstance(inputs, (int, float)) else len(inputs)
        assert ninputs == 1, "only one input supported"
        A = np.eye(n, k=-1)
        B = np.eye(n, 1)*coef
        C = np.eye(1, n, k=n-1)
        D = np.zeros((1,1))
        return ct.ss(A, B, C, D, dt, inputs=inputs, outputs=outputs, **kwargs)
    #l=U[0]*0.8+U[1]*0.6-0.3*ys_[1]
    #expit(l-4)*0.2
    Us = ct.tf([0.6,0.8],[1,0.],dt=dt,inputs='u',outputs='v',name='U')
    delayer1 = time_delay_system(-0.3,dt,dt,inputs='y',outputs='w')
    delayer2 = time_delay_system(1,dt,dt,inputs='y',outputs='yd')
    u_y = ct.summing_junction(inputs=['v','w'],output='d',name='S',dt=dt)
    F3 = ct.NonlinearIOSystem(None,sigmoid_stauration2, inputs='d', outputs='y', dt=dt,name='F')
    sys_wiener3 = ct.interconnect([Us,delayer1,delayer2,u_y,F3],inplist='u', outlist='yd',dt=dt)
    _,ys_,x = ct.input_output_response(sys_wiener3,T,U,x0,return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).flatten().copy()
    return x[0],ys

def simulate_nonlinear6(U,Q,x0=0,seed=None,dt=1):
    #non-linear arx-sigmoid
    N = len(U)
    T = np.arange(1,N+1,dt)
    _,ys_,x = ct.input_output_response(sys_wiener4,T,U,x0,return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).flatten().copy()
    return x[0],ys

def simulate_nonlinear7(U,Q,x0=0,seed=None,dt=1):
    N = len(U)
    T = np.arange(0,1.*dt*(N+2),dt)[:N]

    def vehicle_update(t, x, u, params={}):
        # Set up the system parameters
        m = params.get('m', 1500.)              # vehicle mass, kg
        Cr = params.get('Cr', 0.32)             # resistance coefficient
        F = params.get('F', 2.4)             # frontal area
        rho = params.get('rho', 1.3)            # density of air, kg/m^3

        k = -rho*F*Cr/m/2
    
        return x+dt*u+dt*k*x*abs(x)

    vehicle = ct.NonlinearIOSystem(
        vehicle_update, None, name='vehicle',
        inputs = ('u'), outputs = ('v'), states=('v'),dt=dt)

    _,ys_,x=ct.input_output_response(vehicle,T,U,x0,return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).flatten().copy()
    
    return x[0],ys

def simulate_nonlinear9(U,Q,x0=0,seed=None,dt=1):
    N = len(U)
    T = np.arange(1,N+1,dt)
    def sigmoid3(t,x,u,params):
        offset = params.get('Offset',4)
        scale = params.get('Scale',0.3)
        return expit(scale*u-offset)
        #yhat = 0.5y_{t-1}+0.8*sigmoid(0.3u_{t-1}-4)
    sys_tf = ct.tf([0.,0.5],[1,-0.8],dt=dt)
    sys_hem3 = ct.series(ct.NonlinearIOSystem(None,sigmoid3,inputs=1, outputs=1, dt=dt),sys_tf)
    _,ys_,x = ct.input_output_response(sys_hem3,T,U,x0,return_x=True)
    ys = ys_+ white_noise(T, Q,dt).flatten().copy()
    return x[0],ys


def simulate_nonlinear10(U,Q,x0=[0,0],seed=None,dt=1):
    
    A = np.array([[0.8,1],[-0.1,0.5]])
    B = np.array([1,0])
    C = np.array([1,0])
    D = 0
    sys_ss = ct.ss(A,B,C,D,dt)

    def sigmoid4(t,x,u,params):
        offset = params.get('Offset',4)
        scale = params.get('Scale',0.1)
        return expit(scale*u-offset)
        #x_ = Ax+B*sigmoid(0.2u-4)
    sys_hem4 = ct.series(ct.NonlinearIOSystem(None,sigmoid4,inputs=1, outputs=1, dt=dt),sys_ss)
    
    N = len(U)
    T = np.arange(1,N+1,dt)
    _,ys_,x = ct.input_output_response(sys_hem4,T,U,x0,return_x=True)
    ys = ys_ + white_noise(T, Q,dt,seed=seed).flatten().copy()
    return x,ys

def simulate_nonlinear11(U,Q,x0=[1.,0.5],seed=None,dt=1):
    N = len(U)
    T = np.arange(0,1.*dt*(N+2),dt)[:N]

    def volterra_lodka(t, x, u, params={}):
        # Set up the system parameters
        #dx = 0.4x−0.4xy + u
        #dy = −0.1y+0.2xy
        dx1 = (0.4*x[0]-0.4*x[0]*x[1] + u[0])*dt
        dx2 = (-0.1*x[1]+0.2*x[0]*x[1])*dt
        dx = np.hstack([dx1,dx2])
        return np.maximum(x+dx,np.zeros(2))
    
    def observe(t,x,u,params={}):
        return x
    
    vl = ct.NonlinearIOSystem(
        volterra_lodka, outfcn =observe, name='vl',
        inputs = ('u'), outputs = ('y1','y2'), states=('x1','x2'),dt=dt)
    _,ys_,x=ct.input_output_response(vl,T,U,np.array(x0),return_x=True)
  
    ys = ys_+ white_noise(T, Q,dt,seed=seed).copy()
    return x,ys

def simulate_nonlinear12(U,Q,x0=[.01,.01,0.03],seed=None,dt=1):
    N = len(U)
    T = np.arange(0,1.*dt*(N+2),dt)[:N]

    def two_tanks(t, x, u, params={}):
        A1 = 0.1
        A2 = 0.1
        K = 1
        Cv1 = .2
        Cv2 = .2
        q12 = Cv1*np.sqrt(max(0,x[0]))
        x1 = x[0] + (K*u[0] - q12)*dt/A1
        x2 = x[1] + (q12-Cv2*np.sqrt((max(0,x[1]))))*dt/A2
        x3 = q12
        x_ = np.array([x1,x2,x3])
        return np.maximum(x_,np.zeros(3))
    
    def observe(t,x,u,params={}):
        return x
    
    tt = ct.NonlinearIOSystem(
        two_tanks,observe,name='twotanks',
        inputs = ('u'), outputs = ('y1','y2','y3'), states=('x1','x2','x3'),dt=dt)
    _,ys_,x=ct.input_output_response(tt,T,U,np.array(x0),return_x=True)
    eps = white_noise(T, Q,dt,seed=seed).copy()
    ys = ys_+ eps
    return x,ys

def simulate_nonlinear16(U,Q,x0=[.01,.01],seed=None,dt=1):
    N = len(U)
    T = np.arange(0,1.*dt*(N+2),dt)[:N]

    def two_tanks(t, x, u, params={}):
        A1 = 0.1
        A2 = 0.1
        K = 1
        Cv1 = .2
        Cv2 = .15
        q12 = Cv1*np.sqrt(max(0,x[0]))
        x1 = x[0] + (K*u[0] - q12)*dt/A1
        x2 = x[1] + (q12-Cv2*np.sqrt((max(0,x[1]))))*dt/A2
        x_ = np.array([x1,x2])
        return np.maximum(x_,np.zeros(2))
    
    def observe(t,x,u,params={}):
        return x
    
    tt = ct.NonlinearIOSystem(
        two_tanks,observe,name='twotanks',
        inputs = ('u'), outputs = ('y1','y2'), states=('x1','x2'),dt=dt)
    _,ys_,x=ct.input_output_response(tt,T,U,np.array(x0),return_x=True)
    eps = white_noise(T, Q,dt,seed=seed).copy()
    ys = ys_+ eps
    return x,ys

def simulate_nonlinear13(U,Q,x0=[np.pi,0],seed=None,dt=1):
    #dx1 = x2
    #dx2 = theta1*sin(x1)+theta2*u
    #y = x1
    
    #dx1 = d_angle
    #dx2 = -sin(angle)
    #y = x1
    N = len(U)
    T = np.arange(0,1.*dt*(N+2),dt)[:N]

    def pendulum(t, x, u, params={}):
        theta1 = -1.
        theta2 = 0.1
        dx1 = x[1]*dt
        dx2 = (theta1*np.sin(x[0])+theta2*u[0])*dt
        dx = np.array([dx1,dx2])
        return x+dx
    
    def observe(t,x,u,params={}):
        return x[0]
    
    tt = ct.NonlinearIOSystem(
        pendulum,observe,name='pendulum',
        inputs = ('u'), outputs = ('y'), states=('x1','x2'),dt=dt)
    _,ys_,x=ct.input_output_response(tt,T,U,np.array(x0),return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).copy()
    return x,ys

def simulate_nonlinear17(U,Q,x0=[np.pi,0],seed=None,dt=1):
    #dx1 = x2
    #dx2 = theta1*sin(x1)+theta2*u
    #y = x1
    
    #dx1 = d_angle
    #dx2 = -sin(angle)
    #y = x1
    N = len(U)
    T = np.arange(0,1.*dt*(N+2),dt)[:N]

    def pendulum(t, x, u, params={}):
        theta1 = -24
        theta2 = 1
        dx1 = x[1]*dt
        dx2 = (theta1*np.sin(x[0])+theta2*u[0])*dt
        dx = np.array([dx1,dx2])
        return x+dx
    
    def observe(t,x,u,params={}):
        return x[0]
    
    tt = ct.NonlinearIOSystem(
        pendulum,observe,name='pendulum',
        inputs = ('u'), outputs = ('y'), states=('x1','x2'),dt=dt)
    _,ys_,x=ct.input_output_response(tt,T,U,np.array(x0),return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).copy()
    return x,ys

def simulate_nonlinear18(U,Q,x0=[np.pi,0],seed=None,dt=1):
    #dx1 = x2
    #dx2 = theta1*sin(x1)+theta2*u
    #y = x1
    
    #dx1 = d_angle
    #dx2 = -sin(angle)
    #y = x1
    N = len(U)
    T = np.arange(0,1.*dt*(N+2),dt)[:N]

    def pendulum(t, x, u, params={}):
        theta1 = -24
        theta2 = 0.1
        dx1 = x[1]*dt
        dx2 = (theta1*np.sin(x[0])+theta2*u[0])*dt
        dx = np.array([dx1,dx2])
        return x+dx
    
    def observe(t,x,u,params={}):
        return x[0]
    
    tt = ct.NonlinearIOSystem(
        pendulum,observe,name='pendulum',
        inputs = ('u'), outputs = ('y'), states=('x1','x2'),dt=dt)
    _,ys_,x=ct.input_output_response(tt,T,U,np.array(x0),return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).copy()
    return x,ys

def simulate_nonlinear19(U,Q,x0=[0,0],seed=None,dt=1):
    #dx1 = x2
    #dx2 = theta1*sin(x1)+theta2*u
    #y = x1
    
    #dx1 = d_angle
    #dx2 = -sin(angle)
    #y = x1
    N = len(U)
    T = np.arange(0,1.*dt*(N+2),dt)[:N]

    def pendulum(t, x, u, params={}):
        theta1 = -24
        theta2 = 1
        dx1 = x[1]*dt
        dx2 = (theta1*np.sin(x[0])+theta2*u[0])*dt
        dx = np.array([dx1,dx2])
        return x+dx
    
    def observe(t,x,u,params={}):
        return x[0]
    
    tt = ct.NonlinearIOSystem(
        pendulum,observe,name='pendulum',
        inputs = ('u'), outputs = ('y'), states=('x1','x2'),dt=dt)
    _,ys_,x=ct.input_output_response(tt,T,U,np.array(x0),return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).copy()
    return x,ys

def simulate_linear0(U,Q,x0=0,seed=None,dt=1):
    A = 0.8
    B = 0.5
    C = 1
    D = 0
    sys_ss = ct.ss(A,B,C,D,dt,name='S',inputs='u',outputs='y',states='x')

    N = len(U)
    T = np.arange(1,N+1,dt)
    _,ys_,x = ct.input_output_response(sys_ss,T,U,x0,return_x=True)
    ys = ys_ + white_noise(T, Q,dt,seed=seed).flatten().copy()
    return x,ys

def simulate_linear1(U,Q,x0=0,seed=None,dt=1):
    N = len(U)
    T = np.arange(1,N+1,dt)
    _,ys_,x=ct.input_output_response(sys_fir,T,U,0,return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).flatten().copy()
    return x,ys

def simulate_linear2(U,Q,x0=0,seed=None,dt=1):
    num = [-2,-3.5,-1.1,-0.5,-1]
    denom = [1,1.76,1.84,1.2,0.46,0.11]
    sys_arx = ct.tf(num,denom,dt=dt,name='ARX')
    N = len(U)
    T = np.arange(1,N+1,dt)
    _,ys_,x=ct.input_output_response(ct.tf2io(sys_arx),T,U,0,return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).flatten().copy()
    return x,ys


def simulate_linear3(U,Q,x0=[0,0],seed=None,dt=1):
    
    A = np.array([[0.5,1],[-0.1,0.5]])
    B = np.array([1,0])
    C = np.array([1,0])
    D = 0
    sys_ss = ct.ss(A,B,C,D,dt)

    N = len(U)
    T = np.arange(1,N+1,dt)
    _,ys_,x = ct.input_output_response(sys_ss,T,U,x0,return_x=True)
    ys = ys_ + white_noise(T, Q,dt,seed=seed).flatten().copy()
    return x,ys

def simulate_nonlinear14(U,Q,x0=[0.01],seed=None,dt=1):
    N = len(U)
    T = np.arange(0,1.*dt*(N+2),dt)[:N]

    def vehicle_update(t, x, u, params={}):
        # Set up the system parameters
        m = params.get('m', 1500.)              # vehicle mass, kg
        Cr = params.get('Cr', 0.32)             # resistance coefficient
        F = params.get('F', 2.4)             # frontal area
        rho = params.get('rho', 1.3)            # density of air, kg/m^3

        k = -rho*F*Cr/m/2
    
        return x+dt*np.tanh(u-0.5)+dt*k*(x**2)

    vehicle = ct.NonlinearIOSystem(
        vehicle_update, None, name='vehicle',
        inputs = ('u'), outputs = ('v'), states=('v'),dt=dt)

    _,ys_,x=ct.input_output_response(vehicle,T,U,x0,return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).flatten().copy()
    
    return x[0],ys

def simulate_nonlinear15(U,Q,x0=[0,0],seed=None,dt=1):
    N = len(U)
    T = np.arange(0,1.*dt*(N+2),dt)[:N]

    def weird(t, x, u, params={}):
        # Set up the system parameters
        A = np.array([[0.3,0.5]])
        B = np.array([[0.8]])
        x_ = np.zeros(2)
        x_[:1] = A@x + B@u
        x_[1] = 0.1*np.tanh(x[0]-5)
        
        return x_
    def observe(t,x,u,params={}):
        return x[-1]
        
    sys = ct.NonlinearIOSystem(
        weird, observe, name='weird',
        inputs = ('u'), outputs = ('v'),states=('x1','x2'),dt=dt)

    _,ys_,x=ct.input_output_response(sys,T,U,x0,return_x=True)
    ys = ys_+ white_noise(T, Q,dt,seed=seed).flatten().copy()
    
    return x,ys

def q_alpha_d(dim,alpha=0.05):
    return chi2.ppf(1-alpha,dim)

def scaled_PRBS(N,scale=1,dt=1,seed=None):
    nbits = int(np.ceil(np.log(N)/np.log(2)))
    np.random.seed(14)
    seq = scipy.signal.max_len_seq(nbits=max(15,nbits),state=np.round(np.random.random(15)))[0]
    subseq = np.zeros(N)
    for i in range(N):
        idx=int(np.round(10*(i*dt)-0.5+1e-4))
        subseq[i] = seq[idx]
    return scale*2*(subseq-0.5)

def scaled_PRBS_RA(seed,N,scale=1,dt=1):
    seq = scaled_PRBS(N,scale,dt,14)
    np.random.seed(seed)
    amp = np.random.random(N)
    return seq*amp