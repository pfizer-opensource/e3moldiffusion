import torch
import click
import os

from experiments.utils.data import load_pickle
from experiments.utils.config_file import get_dataset_info


@click.command()
@click.option('--ckpt_path', help='Full path to the checkpoint file.')
@click.option('--save_dir', help='Where the evaluation csv should be saved')
@click.option('--step', help='logging step', default=-1)
@click.option('--server', help='Which server')
@click.option('--type', help='logging step', default="cat")
@click.option('--num_graphs', help='Number of graphs to generate', type=int, default=1000)
@click.option('--batch_size', help='Batch size for generating', type=int, default=100)
@click.option('--device', help='On which device for computation', type=str, default="cuda:0")
def main(ckpt_path, save_dir, step, type, server, num_graphs, batch_size, device):
    device = torch.device(device)
    dataset_info = get_dataset_info("drugs", remove_h=False)
    
    if server == 'delta':
        root = '/home/let55/workspace/projects/e3moldiffusion/experiments/geom/data' # delta
    elif server == 'aws':
        root = '/sharedhome/let55/projects/e3moldiffusion/experiments/geom/data' # aws
    elif server == 'alpha':
        root = '/hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/data' # alpha
    else:
        raise ValueError
    
    train_smiles = load_pickle(os.path.join(root, "processed", "train_smiles.pickle"))
    
    if type == "cont":
        from experiments.geom.train_coordsatomsbonds_x0 import Trainer
        model = Trainer.load_from_checkpoint(ckpt_path,
                                            smiles_list=list(train_smiles),
                                            dataset_info=dataset_info,
                                            strict=False).to(device)
    elif type == "cat":
        from experiments.geom.train_coordsatomsbonds_categorical_x0 import Trainer
        atom_types_distribution = torch.tensor([4.4119e-01, 1.0254e-06, 4.0564e-01, 6.4677e-02, 6.6144e-02, 4.8741e-03,
                                                    0.0000e+00, 9.1150e-07, 1.0847e-04, 1.2260e-02, 4.0306e-03, 0.0000e+00,
                                                    1.0503e-03, 1.9806e-05, 0.0000e+00, 7.5958e-08])
        bond_types_distribution = torch.tensor([9.5523e-01, 3.0681e-02, 2.0021e-03, 4.4172e-05, 1.2045e-02])
        charge_types_distribution = torch.tensor([1.35509982e-06, 1.84150896e-02, 8.86377311e-01, 3.72757628e-02,
                                                    5.79157076e-02, 1.47740195e-05]) # -2, -1, 0, 1, 2, 3
        distributions = {"atoms": atom_types_distribution.float(),
                                     "bonds": bond_types_distribution.float(), 
                                     "charges": charge_types_distribution.float()
                                     }
        model = Trainer.load_from_checkpoint(ckpt_path,
                                            smiles_list=list(train_smiles),
                                            dataset_info=dataset_info,
                                            strict=False, distributions=distributions).to(device)
    else:
        raise ValueError
    

    model.run_evaluation(step=step,
                         dataset_info=model.dataset_info,
                         ngraphs=num_graphs,
                         bs=batch_size,
                         verbose=True,
                         inner_verbose=True,
                         save_dir=save_dir 
                         )
    
if __name__ == '__main__':
    main()