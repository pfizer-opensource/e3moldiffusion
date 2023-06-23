import torch
import click
import os

from experiments.utils.data import load_pickle
from experiments.utils.config_file import get_dataset_info
from experiments.geom.train_coordsatomsbonds_x0 import Trainer


@click.command()
@click.option('--ckpt_path', help='Full path to the checkpoint file.')
@click.option('--num_graphs', help='Number of graphs to generate', type=int, default=1000)
@click.option('--batch_size', help='Batch size for generating', type=int, default=100)
@click.option('--device', help='On which device for computation', type=str, default="cuda:0")
def main(ckpt_path, num_graphs, batch_size, device):
    device = torch.device(device)
    dataset_info = get_dataset_info("drugs", remove_h=False)
    #root = '/home/let55/workspace/projects/e3moldiffusion/experiments/geom/data' # delta
    #root = '/sharedhome/let55/projects/e3moldiffusion/experiments/geom/data' # aws
    root = '/hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/data' # alpha
        
    train_smiles = load_pickle(os.path.join(root, "processed", "train_smiles.pickle"))
    model = Trainer.load_from_checkpoint(ckpt_path,
                                         smiles_list=list(train_smiles),
                                         dataset_info=dataset_info,
                                         strict=False).to(device)
    model.run_evaluation(step=-1,
                         dataset_info=model.dataset_info,
                         ngraphs=num_graphs,
                         bs=batch_size,
                         verbose=True,
                         inner_verbose=True
                         )
    
if __name__ == '__main__':
    main()