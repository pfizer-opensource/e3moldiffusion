import torch
import click
from typing import NamedTuple
from collections import namedtuple

from experiments.data.config_file import get_dataset_info
from experiments.data.data_info import GEOMInfos
from experiments.data.geom_dataset_nonadaptive import GeomDataModule

class TmpCfg():
    def __init__(self) -> None:
        self.remove_hs = False
        
              
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
    
    datamodule = GeomDataModule(root=root,
                                batch_size=1,
                                num_workers=1,
                                pin_memory=True,
                                persistent_workers=True,
                                with_hydrogen=True
                                )
    datamodule.prepare_data()
    datamodule.setup("fit")

    hparams = TmpCfg()
        
    dataset_statistics = GEOMInfos(datamodule, hparams)
    dataset_info = get_dataset_info("drugs", remove_h=False)
    train_smiles = datamodule.train_dataset.smiles
    
    
    if type == "cont":
        from experiments.diffusion_continuous import Trainer
       
    elif type == "cat":
        from experiments.diffusion_discrete import Trainer
    else:
        raise ValueError
    

    model = Trainer.load_from_checkpoint(ckpt_path,
                                         dataset_info=dataset_info,
                                         dataset_statistics=dataset_statistics,
                                         smiles_list=list(train_smiles)).to(device)
    
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