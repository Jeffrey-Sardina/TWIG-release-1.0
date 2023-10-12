import sys
from load_data import do_load
from twig_nn import NeuralNetwork_HPs_v1, NeuralNetwork_HPs_v2
from trainer import run_training
import glob
import os
import torch

'''
===============
Reproducibility
===============
'''
torch.manual_seed(42)

def load_nn(version, checkpoint_dir, model_name_prefix, first_epochs, second_epochs):
    print('loading NN')

    # check if there is a model to restore
    model = None
    if checkpoint_dir is not None:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
        checkpoints = [c for c in checkpoints if model_name_prefix in c]
        if len(checkpoints) > 0:
            checkpoints.sort(key=lambda name: int(os.path.basename(name).split('_')[0]))
            latest_checkpoint = os.path.basename(checkpoints[-1])
            print(f'loading from checkpoint {latest_checkpoint}')
            epoch_data = latest_checkpoint.replace('*.pt', '').split('_')[-1] #in form eA_eB
            epoch_data = epoch_data.replace('e', '').replace('.pt', '').split('-')
            first_epochs_data, second_epochs_data = int(epoch_data[0]), int(epoch_data[1])
            first_epochs -= first_epochs_data
            second_epochs -= second_epochs_data
            model = torch.load(os.path.join(checkpoint_dir, latest_checkpoint))
        else:
            print('no checkpoint found; creating model from scratch')

    supported_versions = [1, 2]
    assert version in supported_versions
    if version == 1:
        if model is None:
            model = NeuralNetwork_HPs_v1(
                n=23 + 9
            )
        layers_to_freeze = [
            model.linear1,
            model.linear2
        ]
    elif version == 2:
        if model is None:
            model = NeuralNetwork_HPs_v2(
                n_struct = 23,
                n_hps = 9
            )
        layers_to_freeze = [
            model.linear_struct_1,
            model.linear_struct_2,
            model.linear_hps_1,
            model.linear_integrate_1
        ]
    print("done loading NN")
    return model, layers_to_freeze, first_epochs, second_epochs

def load_dataset(dataset_name):
    print('loading dataset')
    supported_datasets = ['UMLS', 'Nations']
    assert dataset_name in supported_datasets
    if dataset_name == 'UMLS':
        load_ids = ('2.1', '2.2', '2.3', '2.4')
    elif dataset_name == 'Nations':
        load_ids = ('2', '2.1', '2.2', '2.3', '2.4', '2.5')
    print(load_ids)
    training_dataloader, testing_dataloader = do_load(dataset_name, load_ids)
    print("done loading dataset")
    return training_dataloader, testing_dataloader

def train_and_eval(
        model,
        training_dataloader,
        testing_dataloader,
        layers_to_freeze,
        first_epochs,
        second_epochs,
        lr,
        verbose=True,
        model_name_prefix='model',
        checkpoint_dir='checkpoints/',
        checkpoint_every_n=5
    ):
    print("running training and eval")
    r2_mrr, test_loss = run_training(model,
        training_dataloader,
        testing_dataloader,
        layers_to_freeze,
        first_epochs=first_epochs,
        second_epochs=second_epochs,
        lr=lr,
        verbose=verbose,
        model_name_prefix=model_name_prefix,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_n=checkpoint_every_n
    )
    print("done with training and eval")
    return r2_mrr, test_loss

def main(version, dataset_name, first_epochs, second_epochs):
    checkpoint_dir = 'checkpoints/'
    model_name_prefix = f'v{version}_{dataset_name}'
    model, layers_to_freeze, first_epochs, second_epochs = load_nn(
        version,
        checkpoint_dir,
        model_name_prefix,
        first_epochs,
        second_epochs
    )

    training_dataloader, testing_dataloader = load_dataset(dataset_name
                                                           )
    r2_mrr, test_loss = train_and_eval(
        model,
        training_dataloader,
        testing_dataloader,
        layers_to_freeze,
        first_epochs=first_epochs,
        second_epochs=second_epochs,
        lr=5e-3,
        verbose = True,
        model_name_prefix=model_name_prefix,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_n=1
    )
    return r2_mrr, test_loss

if __name__ == '__main__':
    version = int(sys.argv[1])
    dataset_name = sys.argv[2]
    first_epochs = int(sys.argv[3])
    second_epochs = int(sys.argv[4])
    main(version, dataset_name, first_epochs, second_epochs)
