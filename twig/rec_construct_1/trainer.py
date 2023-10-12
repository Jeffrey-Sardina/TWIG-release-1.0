import torch
from torch import nn
from torcheval.metrics.functional import r2_score
import torch.nn.functional as F
import os

'''
===============
Reproducibility
===============
'''
torch.manual_seed(42)

'''
====================
Constant Definitions
====================
'''
device = "cuda"

'''
================
Helper Functions
================
'''
def d_hist(X, n_bins, min_val, max_val):
    # n_elems = torch.prod(torch.tensor(X.shape))
    bins = torch.linspace(start=min_val, end=max_val, steps=n_bins+1)[1:]
    freqs = torch.zeros(size=(n_bins,)).to(device)
    last_val = None
    sharpness = 1
    for i, curr_val in enumerate(bins):
        if i == 0:
            count = F.sigmoid(sharpness * (curr_val - X))
        elif i == len(bins) - 1:
            count = F.sigmoid(sharpness * (X - last_val))
        else:
            count = F.sigmoid(sharpness * (X - last_val)) \
                * F.sigmoid(sharpness * (curr_val - X))
        count = torch.sum(count)
        # freqs[i] += (count + 1) / (n_elems + n_bins) # +1, +n_bins since if a count is 0, we need it to be 1 instead
        freqs[i] += (count + 1) #new; +1 to avoid 0s as this will be logged
        last_val = curr_val
    freqs = freqs / torch.sum(freqs) #new
    return freqs

def train_epoch(dataloader,
            model,
            mrr_loss,
            unordered_rank_list_loss,
            optimizer,
            alpha=1,
            gamma=1,
            n_bins=30,
            verbose=False):
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        R_true = y.to(device)
        mrr_true = torch.mean(1 / R_true)
        R_pred, mrr_pred = model(X)

        # get dists
        min_val = float(torch.min(R_true))
        max_val = float(torch.max(R_true))
        
        R_true_dist = d_hist(
            R_true,
            n_bins=n_bins,
            min_val=min_val,
            max_val=max_val
        )
        R_pred_dist = d_hist(
            R_pred,
            n_bins=n_bins,
            min_val=min_val,
            max_val=max_val
        )

        # compute loss
        mrrl = mrr_loss(mrr_pred, mrr_true)
        urll = unordered_rank_list_loss(R_pred_dist.log(), R_true_dist) #https://discuss.pytorch.org/t/kl-divergence-produces-negative-values/16791/5
        loss = alpha * mrrl + gamma * urll
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 500 == 0 and verbose:
            print(f"batch {batch} / {num_batches} mrrl: {alpha * mrrl.item():>4f}; urll: {gamma * urll.item():>4f}; ")

def test(dataloader,
            model,
            mrr_loss,
            unordered_rank_list_loss,
            alpha=1,
            gamma=1,
            n_bins=30,
            verbose=False):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    mrr_preds = []
    mrr_trues = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            if batch % 500 == 0 and verbose:
                print(f'Testing: batch {batch} / {num_batches}')
            X = X.to(device)
            R_true = y.to(device)
            mrr_true = torch.mean(1 / R_true)
            R_pred, mrr_pred = model(X)

            # get dists
            min_val = float(torch.min(R_true))
            max_val = float(torch.max(R_true))
            
            R_true_dist = d_hist(
                R_true,
                n_bins=n_bins,
                min_val=min_val,
                max_val=max_val
            )
            R_pred_dist = d_hist(
                R_pred,
                n_bins=n_bins,
                min_val=min_val,
                max_val=max_val
            )

            # compute loss
            mrrl = mrr_loss(mrr_pred, mrr_true)
            urll = unordered_rank_list_loss(R_pred_dist.log(), R_true_dist) #https://discuss.pytorch.org/t/kl-divergence-produces-negative-values/16791/5
            loss = alpha * mrrl + gamma * urll
            
            test_loss += loss.item()
            mrr_preds.append(mrr_pred)
            mrr_trues.append(mrr_true)
    
    # are these getting and staying sorted somehow? 
    r2_ranks = r2_score(
        R_pred,
        R_true
    )
    print(f'ranks r2 : {r2_ranks}')
    r2_ranks_sort = r2_score(
        torch.sort(R_pred)[0],
        torch.sort(R_true)[0]
    )
    print(f'ranks r2 (sorted) : {r2_ranks_sort}')
    print(f'ranks R: {torch.corrcoef(torch.stack([R_pred.flatten(), R_true.flatten()]))}')

    r2_mrr = None
    if len(mrr_preds) > 1:
        r2_mrr = r2_score(
            torch.tensor(mrr_preds),
            torch.tensor(mrr_trues),
        )
        print(f'r_mrr = {torch.corrcoef(torch.tensor([mrr_preds, mrr_trues]))}')
        print(f'r2_mrr = {r2_mrr}')
    else:
        print([round(float(x), 2) for x in mrr_preds])
        print([round(float(x), 2) for x in mrr_trues])

    test_loss /= num_batches  
    print(f"test_loss: {test_loss}")

    return r2_mrr, test_loss

def run_training(
        model,
        training_dataloader,
        testing_dataloader,
        layers_to_freeze=[],
        first_epochs = 30,
        second_epochs = 60,
        lr=5e-3,
        verbose = True,
        model_name_prefix="model",
        checkpoint_dir='checkpoints/',
        checkpoint_every_n=5
    ):
    model.to(device)
    mrr_loss = nn.MSELoss()
    unordered_rank_list_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)

    # Training
    model.train()
    print(f'REC: Training with epochs in stages 1: {first_epochs} and 2: {second_epochs}')
    curr_checkpoint_num = 1

    alpha = 0
    gamma = 1
    for layer in layers_to_freeze:
        layer.requires_grad_ = True
    for t in range(first_epochs):
        print(f"Epoch {t+1} -- ", end='')
        train_epoch(training_dataloader,
                model,
                mrr_loss,
                unordered_rank_list_loss,
                optimizer, 
                alpha=alpha,
                gamma=gamma,
                verbose=verbose
            )
        if (t+1) % checkpoint_every_n == 0:
            print(f'Saving checkpoint at [1] epoch {t+1}')
            state_data = f'e{t+1}-e0'
            torch.save(
                model,
                    os.path.join(
                        checkpoint_dir,
                        f'{curr_checkpoint_num}_{model_name_prefix}_{state_data}.pt'
                    )
                )
            curr_checkpoint_num += 1
    print("Done Training (dist)!")

    alpha = 10
    gamma = 1
    for layer in layers_to_freeze:
        layer.requires_grad_ = False
    for t in range(second_epochs):
        print(f"Epoch {t+1} -- ", end='')
        train_epoch(training_dataloader,
                model,
                mrr_loss,
                unordered_rank_list_loss,
                optimizer, 
                alpha=alpha,
                gamma=gamma,
                verbose=verbose
            )
        if (t+1) % checkpoint_every_n == 0:
            print(f'Saving checkpoint at [2] epoch {t+1}')
            state_data = f'e{first_epochs}-e{t+1}'
            torch.save(
                model,
                    os.path.join(
                        checkpoint_dir,
                        f'{curr_checkpoint_num}_{model_name_prefix}_{state_data}.pt'
                    )
                )
            curr_checkpoint_num += 1
    print("Done Training (mrr)!")

    # Testing
    model.eval()
    print(f'REC: Testing model')
    r2_mrr, test_loss = test(testing_dataloader,
        model,
        mrr_loss,
        unordered_rank_list_loss,
        alpha=alpha,
        gamma=gamma,
        verbose=verbose
    )
    print("Done Testing!")
    print(f'REC: Results: r2_mrr = {r2_mrr}; test_loss = {test_loss}')

    return r2_mrr, test_loss
