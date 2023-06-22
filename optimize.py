from datetime import datetime
from functools import partial  # printf
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from vendi_torch import score_K  # type: ignore[import]


printf = partial(print, flush=True)


""" Methods """
def weights_to_prob(w, log_in=True, log_out=True):  # Normalize
    if log_in:  # Input is in log space
        w = torch.exp(w)  # to linear space
    p = w / w.sum()  # probabilities
    if log_out:  # Output should be log space
        p = torch.log(p)  # to log space
    return p

def entropy(w):  # Shannon entropy
    p = weights_to_prob(w, log_out=False)  # probabilities
    return -torch.sum(p * torch.log(p))

def energy(w):  # Onicescu’s informational energy
    p = weights_to_prob(w, log_out=False)  # probabilities
    return torch.sum(torch.square(p))

def diversity_criterion(K, w, q=1):  # Maximise diversity score
    p = weights_to_prob(w)  # probabilities
    return score_K(K, q=q, p=p, p_log=True)

def probability_criterion(w):  # Keep sum of probabilities close to one
    p = weights_to_prob(w)  # probabilities
    return p.exp().sum() - 1.


""" Optimization """
# Log version, Adam optimizer
def optimize_weights(K, 
        q=1,  # Vendi Score parameter
        dataset="n/a",
        num_steps=100,  # Number of optimization steps
        lr=0.1,  # Learning rate
        lr_decay=0.99,  # Coefficient for learning rate decay
        lr_step=5,  # Number of steps to decrease the learning rate
        loss_balance=0.5,  # Loss balance (diversity and entropy) hyperparameter

        log_step=10,  # Number of steps to log
        log_prec=6,  # Log floating point precision

        save_step=0,  # Number of steps to save, 0 for smart saving
        save_path='save',  # Directory for saving
        save_name='diversity_weights',  # Prefix of weights file to save

        seed=None,
        device=torch.device('cpu'),
        dtype=torch.float64):
    """ Setup """
    run_id = datetime.now().strftime('%Y%m%d%H%M%S')  # YYYYMMDDHHMMSS

    num_samples = K.shape[0]  # Number of samples

    if seed:
        torch.manual_seed(seed)

    config = {  # Collection of important settings
        'dataset': dataset, 'num_samples': num_samples, 'num_steps': num_steps, 
        'lr': lr, 'lr_decay': lr_decay, 'lr_step': lr_step,
        'loss_balance': loss_balance,
        'seed': seed,'device': str(device), 'dtype': dtype
    }

    # Saving
    save_path = Path(f"{save_path}-{run_id}")
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path/"config.txt", "w") as f:
        for setting in config:
            f.write(f"{setting}: {config[setting]}\n")

    # Initialize
    weight_init = 0  # Initial weights: 0 in log space (1 in lin space)
    weights = torch.full((num_samples,), torch.tensor(weight_init),
                         dtype=dtype, device=device, 
                         requires_grad=True)
    info = {'step': -1, 'vendi_score': 0, 'entropy_score': 0, 'prob_error': 1}

    # Optimization algorithm
    optimizer = Adam([weights], lr=lr)
    # Decay learning rate over steps
    scheduler = StepLR(optimizer, lr_step, gamma=lr_decay)

    printf("Step", '\t',
           "Vendi score ↑", '\t',
           "Entropy ↑    ", '\t',
           "Prob error ↓ ", '\t',
           "Learning rate", '\t',
           "Saved")
    for step in range(1, num_steps + 1):
        lr_update = scheduler.get_last_lr()[0]  # Get current learning rate

        diversity_loss = diversity_criterion(K, weights, q)
        entropy_loss = entropy(weights)

        optimizer.zero_grad(set_to_none=True)
        loss = -(loss_balance * diversity_loss + (1.-loss_balance) * entropy_loss)
        loss.backward()
        optimizer.step()

        # Loss values for logging
        vendi_score = diversity_loss.item()
        entropy_score = entropy_loss.item()
        prob_error = probability_criterion(weights).item()

        """ Save """
        saved = False  # Boolean flag, save this step?
        if save_step > 0:
            # Simple save based on step count
            if step % save_step == 0:
                saved = True
        elif (vendi_score > info['vendi_score']) and (entropy_score > info['entropy_score']):
            # Smart save based on losses
            saved = True
        
        if saved:
            save_file = save_path/f"{save_name}-step{step}.npz"
            info = {
                'step': step,
                'vendi_score': vendi_score,
                'entropy_score': entropy_score,
                'prob_error': prob_error
            }
            np.savez(save_file, 
                     info=info, 
                     config=config, 
                     weights=weights.detach().cpu().numpy())

        """ Print report """
        if (step % log_step == 0) or saved:
            printf(step, '\t',
                   f"{vendi_score:.{log_prec}f}", '\t',
                   f"{entropy_score:.{log_prec}f}", '\t',
                   f"{prob_error:.{log_prec}f}", '\t',
                   f"{lr_update:.{log_prec}f}", '\t',
                   "*" if saved else "")

        """ Learning rate decay """
        scheduler.step()  # Decrease learning rate

    printf(f"Final VS: {vendi_score:.{log_prec}f}")
    return weights
