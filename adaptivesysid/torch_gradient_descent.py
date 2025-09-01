import numpy as np
import torch
from torch.optim import AdamW,Adam,SGD

def step(fun,opt,scheduler,var,args):
    opt.zero_grad()
    loss = fun(var,*args)
    loss.backward()
    
    #torch.nn.utils.clip_grad_norm_([var], max_norm=5.0)
    
    opt.step()
    scheduler.step()
    with torch.no_grad():
        loss_val = loss.item()
    return loss_val

def step_minimax(funs,opt_desc,opt_asc,var_desc,var_asc,args):
    fun = funs[0]
        
    opt_desc.zero_grad()
    opt_asc.zero_grad()
    
    loss = fun(var_desc,var_asc,*args)
    loss.backward()
    
    opt_desc.step()
    opt_asc.step()
    
    with torch.no_grad():
        loss_val = loss.item()
    return loss_val


def minimize(fun,init_val,kparams,n_iter=3000,lr=1e-2,early_stop_tol=1e-4,early_stop_patience=10):
    torch.set_default_tensor_type(torch.DoubleTensor)
    losses = np.empty(n_iter)
    
    #Descending optimizer
    var = torch.tensor(init_val).requires_grad_(True)
    #opt = AdamW([var],lr = lr) 
    
    opt = torch.optim.AdamW([var], lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

    
    min_loss = step(fun,opt,scheduler,var,kparams)
    bestvar = var.clone().detach()
    stopped_early = False
    patience_counter = 0
    for i in range (n_iter):
        loss_val = step(fun,opt,scheduler,var,kparams)
        
        if np.isnan(loss_val)==False:
            losses[i] = loss_val
        
         # --- Early Stopping ---
        if  np.isnan(loss_val)==False:
            if loss_val < min_loss - early_stop_tol:
                min_loss = loss_val
                bestvar = var.clone().detach()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    stopped_early = True
                    break
        else:
            break
            
    return bestvar,losses[:i]