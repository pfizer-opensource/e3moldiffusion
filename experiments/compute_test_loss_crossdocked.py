import numpy as np
import torch
import os
import os.path as osp
import pickle
import argparse
from argparse import ArgumentParser
from tqdm import tqdm
import random
from experiments.data.data_info import GeneralInfos as DataInfos
from experiments.data.data_info import full_atom_encoder as atom_encoder
from experiments.data.ligand.ligand_dataset_nonadaptive import (
    LigandPocketDataModule as DataModule,
)
from experiments.diffusion_discrete_pocket import Trainer

class TmpCfg:
    def __init__(self, dataset_root, batch_size) -> None:
        self.remove_hs = True
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = 1
        self.dataset = 'crossdocked'
        self.joint_property_prediction = False
        self.regression_property = "none"
        self.property_training = False
        self.num_bond_classes = 5
        self.num_charge_classes = 6
        

def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    cfg = TmpCfg(dataset_root=args.dataset_root, batch_size=args.batch_size)
    datamodule = DataModule(cfg)
    dataset_info = DataInfos(datamodule=datamodule, cfg=cfg)
    histogram = os.path.join(
                            cfg.dataset_root, "size_distribution.npy"
                        )
    device = torch.device(args.device)
    histogram = np.load(histogram).tolist()
    test_loader = datamodule.test_dataloader()
    model = Trainer.load_from_checkpoint(args.model_path,
                                     histogram=histogram,
                                     dataset_info=dataset_info,
                                     smiles_list=datamodule.train_dataset.smiles,
                                     map_location=device).to(device)
    
    atoms_losses_batch = []
    coords_losses_batch = []
    bonds_losses_batch = []

    print(f"Processing {len(test_loader)} batches.")
    for i, data in enumerate(test_loader):
        
        data = data.to(device)
        atoms_losses = []
        coords_losses = []
        bonds_losses = []
        
        for t in tqdm(range(1, model.hparams.timesteps + 1)):
            at_loss = 0.0
            co_loss = 0.0
            bo_loss = 0.0
            for j in range(args.repetitions):
                t_t = torch.ones((args.batch_size,), dtype=torch.long, device=device) * t
                with torch.no_grad():
                    out_dict = model(data, t_t)
                
                weights = None
                true_data = {
                            "coords": (
                                out_dict["coords_true"]
                                if model.hparams.continuous_param == "data"
                                else out_dict["coords_noise_true"]
                            ),
                            "atoms": out_dict["atoms_true"],
                            "charges": out_dict["charges_true"],
                            "bonds": out_dict["bonds_true"],
                        }
                
                coords_pred = out_dict["coords_pred"]
                atoms_pred = out_dict["atoms_pred"]
                atoms_pred, charges_pred = atoms_pred.split(
                    [model.num_atom_types, model.num_charge_classes], dim=-1
                )
                edges_pred = out_dict["bonds_pred"]
                pred_data = {
                    "coords": coords_pred,
                    "atoms": atoms_pred,
                    "charges": charges_pred,
                    "bonds": edges_pred,
                }
                true_data["properties"] = None
                pred_data["properties"] = None
                
                loss = model.diffusion_loss(
                            true_data=true_data,
                            pred_data=pred_data,
                            batch=data.pos_batch,
                            bond_aggregation_index=out_dict["bond_aggregation_index"],
                            intermediate_coords=False,
                            weights=weights,
                            molsize_weights=None,
                        )
                at_loss += loss["atoms"].item()
                co_loss += loss["coords"].item()
                bo_loss += loss["bonds"].item()
        
        
            at_loss /= args.repetitions
            co_loss /= args.repetitions
            bo_loss /= args.repetitions
        
            atoms_losses.append(at_loss)
            coords_losses.append(co_loss)
            bonds_losses.append(bo_loss)
            
        atoms_losses_batch.append(atoms_losses)
        coords_losses_batch.append(coords_losses)
        bonds_losses_batch.append(bonds_losses)

    
    atoms_losses_batch = np.array(atoms_losses_batch)
    coords_losses_batch = np.array(coords_losses_batch)
    bonds_losses_batch = np.array(bonds_losses_batch)
    
    print("atom loss mean: ", np.mean(atoms_losses_batch))
    print("coods loss mean: ", np.mean(coords_losses_batch))
    print("bonds loss mean: ", np.mean(bonds_losses_batch))
    
    basedir = osp.dirname(args.model_path)
    np.save(osp.join(basedir, f"atoms_losses_seed{args.seed}.npy"), atoms_losses_batch)
    np.save(osp.join(basedir, f"coords_losses_seed{args.seed}.npy"), coords_losses_batch)
    np.save(osp.join(basedir, f"bonds_losses_seed{args.seed}.npy"), bonds_losses_batch)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Test set loss calculation on CrossDocked2020')
    parser.add_argument("--model-path",
                        default="/scratch1/e3moldiffusion/logs/crossdocked/from_gamma/new_7A_td_data_12L_sv256_e128_cutoff10A_new_rbf_global_edge_interaction_hybrid_knn32/best_valid.ckpt",
                        type=str,)
    parser.add_argument("--dataset-root",
                        default="/scratch1/e3moldiffusion/data/crossdocked/crossdocked_noH_cutoff7_TargetDiff_atmass/",
                        type=str,)
    parser.add_argument("--batch-size",
                        default=16,
                        type=int,)
    parser.add_argument("--repetitions",
                        default=2,
                        type=int,)
    parser.add_argument("--seed",
                        default=0,
                        type=int,)
    parser.add_argument("--device",
                        default="cuda",
                        type=str,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)