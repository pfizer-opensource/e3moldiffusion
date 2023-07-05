import json
import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         ModelSummary, TQDMProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean, scatter_add
from torch_sparse import coalesce
from tqdm import tqdm

from callbacks.ema import ExponentialMovingAverage
from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from e3moldiffusion.molfeat import atom_type_config, get_bond_feature_dims
from e3moldiffusion.sde import DiscreteDDPM
from e3moldiffusion.categorical import CategoricalDiffusionKernel

from experiments.utils.data import load_pickle
from experiments.utils.config_file import get_dataset_info
from experiments.utils.config_file import get_dataset_info
#from experiments.utils.sampling import (Molecule,
#                                        analyze_stability_for_molecules)
from experiments.utils.analyze import Molecule, analyze_stability_for_molecules
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss

logging.getLogger("lightning").setLevel(logging.WARNING)

def get_num_atom_types_geom(dataset: str):
    assert dataset in ["qm9", "drugs"]
    return len(atom_type_config(dataset=dataset))


def zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0):
    out = x - scatter_mean(x, index=batch, dim=dim, dim_size=dim_size)[batch]
    return out

def assert_zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0, eps: float = 1e-6):
    out = scatter_mean(x, index=batch, dim=dim, dim_size=dim_size).mean()
    return abs(out) < eps

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]


class Trainer(pl.LightningModule):
    def __init__(self,
                 hparams: dict,
                 dataset_info: dict,
                 smiles_list: list,
                 distributions: Optional[Dict] = None
                 ):
        super().__init__()
        self.save_hyperparameters(hparams)

        if distributions is None:
            atom_types_distribution = torch.tensor([4.4119e-01, 1.0254e-06, 4.0564e-01, 6.4677e-02, 6.6144e-02, 4.8741e-03,
                                                    0.0000e+00, 9.1150e-07, 1.0847e-04, 1.2260e-02, 4.0306e-03, 0.0000e+00,
                                                    1.0503e-03, 1.9806e-05, 0.0000e+00, 7.5958e-08])
            bond_types_distribution = torch.tensor([9.5523e-01, 3.0681e-02, 2.0021e-03, 4.4172e-05, 1.2045e-02])
            charge_types_distribution = torch.tensor([1.35509982e-06, 1.84150896e-02, 8.86377311e-01, 3.72757628e-02,
                                                      5.79157076e-02, 1.47740195e-05]) # -2, -1, 0, 1, 2, 3
        else:
            atom_types_distribution = distributions.get("atoms")
            bond_types_distribution = distributions.get("bonds")
            charge_types_distribution = distributions.get("charges")
        
        self.register_buffer('atoms_prior', atom_types_distribution.clone())
        self.register_buffer('bonds_prior', bond_types_distribution.clone())
        self.register_buffer('charges_prior', charge_types_distribution.clone())
        
        self.hparams.num_atom_types = get_num_atom_types_geom(dataset="drugs")
        self.hparams.num_charge_classes = 6
        if hparams.get('no_h'):
            print("Training without hydrogen")
            self.hparams.num_atom_types -= 1
            
        self.hparams.num_bond_types = 5
        self.smiles_list = smiles_list
        self.num_atom_features = self.hparams.num_atom_types
        self.num_bond_classes = 5
        self.num_charge_classes = 6
        
        self.dataset_info = dataset_info

        self.i = 0
            
        empirical_num_nodes = self._get_empirical_num_nodes()
        self.register_buffer(name='empirical_num_nodes', tensor=empirical_num_nodes)

        
        self.valency_pred = False
        self.model = DenoisingEdgeNetwork(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            num_layers=hparams["num_layers"],
            use_norm=not hparams["omit_norm"],
            use_cross_product=not hparams["omit_cross_product"],
            num_atom_types=self.num_atom_features + self.num_charge_classes,
            num_bond_types=self.num_bond_classes,
            edge_dim=hparams['edim'],
            cutoff_local=hparams["cutoff_local"],
            rbf_dim=hparams["rbf_dim"],
            vector_aggr=hparams["vector_aggr"],
            fully_connected=hparams["fully_connected"],
            local_global_model=False,
            recompute_edge_attributes=True,
            recompute_radius_graph=False
        )

        self.sde = DiscreteDDPM(beta_min=hparams["beta_min"],
                                beta_max=hparams["beta_max"],
                                N=hparams["timesteps"],
                                scaled_reverse_posterior_sigma=True,
                                schedule="cosine",
                                enforce_zero_terminal_snr=False)
        
        self.cat_atoms = CategoricalDiffusionKernel(terminal_distribution=atom_types_distribution,
                                                    alphas=self.sde.alphas.clone()
                                                    )
        self.cat_bonds = CategoricalDiffusionKernel(terminal_distribution=bond_types_distribution,
                                                    alphas=self.sde.alphas.clone()
                                                    )
        self.cat_charges = CategoricalDiffusionKernel(terminal_distribution=charge_types_distribution,
                                                    alphas=self.sde.alphas.clone()
                                                    )
        
        self.mse_loss = MSELoss()
        self.ce_loss = CrossEntropyLoss()
        
        
                 
    def _get_empirical_num_nodes(self):
        if not self.hparams.no_h:
            pp = '/home/let55/workspace/projects/e3moldiffusion/experiments/'   # delta
            pp = '/sharedhome/let55/projects/e3moldiffusion/experiments/'  # aws
            pp = '/hpfs/userws/let55/projects/e3moldiffusion/experiments/' # alpha
            with open(f'{pp}geom/num_nodes_geom_midi.json', 'r') as f:
                num_nodes_dict = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
        else:
            with open(f'{pp}geom/num_nodes_geom_midi_no_h.json', 'r') as f:
                num_nodes_dict = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
                
        num_nodes_dict = self.dataset_info.get('n_nodes')
        max_num_nodes = max(num_nodes_dict.keys())
        empirical_distribution_num_nodes = {i: num_nodes_dict.get(i) for i in range(max_num_nodes)}
        empirical_distribution_num_nodes_tensor = {}

        for key, value in empirical_distribution_num_nodes.items():
            if value is None:
                value = 0
            empirical_distribution_num_nodes_tensor[key] = value
        empirical_distribution_num_nodes_tensor = torch.tensor(list(empirical_distribution_num_nodes_tensor.values())).float()
        return empirical_distribution_num_nodes_tensor
                
    def get_list_of_edge_adjs(self, edge_attrs_dense, batch_num_nodes):
        ptr = torch.cat([torch.zeros(1, device=batch_num_nodes.device, dtype=torch.long), batch_num_nodes.cumsum(0)])
        edge_tensor_lists = []
        for i in range(len(ptr) - 1):
            select_slice = slice(ptr[i].item(), ptr[i+1].item())
            e = edge_attrs_dense[select_slice, select_slice]
            edge_tensor_lists.append(e)
        return edge_tensor_lists

    @torch.no_grad()
    def generate_graphs(self,
                        num_graphs: int,
                        empirical_distribution_num_nodes: torch.Tensor,
                        device: torch.device,
                        verbose=False,
                        save_traj=False):
        
        pos, atom_types, charge_types, edge_types,\
        edge_index_global, batch_num_nodes, trajs = self.reverse_sampling(num_graphs=num_graphs,
                                                                        device=device,
                                                                        empirical_distribution_num_nodes=empirical_distribution_num_nodes,
                                                                        verbose=verbose,
                                                                        save_traj=save_traj)

        pos_splits = pos.detach().cpu().split(batch_num_nodes.cpu().tolist(), dim=0)
        
        charge_types_integer = torch.argmax(charge_types, dim=-1)
        # offset back
        charge_types_integer = charge_types_integer - 2
        charge_types_integer_split = charge_types_integer.detach().cpu().split(batch_num_nodes.cpu().tolist(), dim=0)
        atom_types_integer = torch.argmax(atom_types, dim=-1)
        if self.hparams.no_h:
            raise NotImplementedError # remove in future or implement
            atom_types_integer += 1
            
        atom_types_integer_split = atom_types_integer.detach().cpu().split(batch_num_nodes.cpu().tolist(), dim=0)
        
        return pos_splits, atom_types_integer_split, charge_types_integer_split, edge_types, edge_index_global, batch_num_nodes, trajs    
    
    
    @torch.no_grad()
    def run_evaluation(self,
                       step: int,
                       dataset_info,
                       ngraphs: int = 4000,
                       bs: int = 500,
                       save_dir: str = None,
                       verbose: bool = False,
                       inner_verbose=False):
        b = ngraphs // bs
        l = [bs] * b
        if sum(l) != ngraphs:
            l.append(ngraphs - sum(l))
        assert sum(l) == ngraphs
        
        molecule_list = []
        if verbose:
            print(f"Creating {ngraphs} graphs in {l} batches")
        for _, num_graphs in enumerate(l):
            start = datetime.now()
            pos_splits, atom_types_integer_split, charge_types_integer_split, \
            edge_types, edge_index_global, batch_num_nodes, _ = self.generate_graphs(
                                                                                     num_graphs=num_graphs,
                                                                                     verbose=inner_verbose,
                                                                                     device=self.empirical_num_nodes.device,
                                                                                     empirical_distribution_num_nodes=self.empirical_num_nodes,
                                                                                     save_traj=False)
            
            n = batch_num_nodes.sum().item()
            edge_attrs_dense = torch.zeros(size=(n,n,5), dtype=edge_types.dtype, device=edge_types.device)
            edge_attrs_dense[edge_index_global[0, :], edge_index_global[1, :], :] = edge_types
            edge_attrs_dense = edge_attrs_dense.argmax(-1)
            edge_attrs_splits = self.get_list_of_edge_adjs(edge_attrs_dense, batch_num_nodes)
            
            for positions, atom_types, charges, edges in zip(pos_splits,
                                                    atom_types_integer_split,
                                                    charge_types_integer_split,
                                                    edge_attrs_splits):
                molecule = Molecule(atom_types=atom_types, positions=positions,
                                    dataset_info=dataset_info,
                                    charges=charges,
                                    bond_types=edges
                                    )
                molecule_list.append(molecule)
            
        
        stability_dict, validity_dict, all_generated_smiles = analyze_stability_for_molecules(molecule_list=molecule_list, 
                                                                                            dataset_info=dataset_info,
                                                                                            smiles_train=self.smiles_list,
                                                                                            bonds_given=True
                                                                                            )

        if verbose:
            print(f'Run time={datetime.now() - start}')
        total_res = dict(stability_dict)
        total_res.update(validity_dict)
        print(total_res)
        total_res = pd.DataFrame.from_dict([total_res])        
        print(total_res)
        
        total_res['step'] = step
        total_res['epoch'] = self.current_epoch
        try:
            if save_dir is None:
                save_dir = os.path.join(self.hparams.save_dir, 'run' + self.hparams.id, 'evaluation.csv')
            else:
                save_dir = os.path.join(save_dir, 'evaluation.csv')
            with open(save_dir, 'a') as f:
                total_res.to_csv(f, header=True)
        except:
            pass
        return total_res
        
    def on_validation_epoch_end(self, *args, **kwargs):
        
        if (self.current_epoch + 1) % self.hparams.test_interval == 0:  
            print(f"Running evaluation in epoch {self.current_epoch + 1}")      
            final_res = self.run_evaluation(step=self.i, dataset_info=self.dataset_info,
                                            ngraphs=1000, bs=self.hparams.batch_size,
                                            verbose=True, inner_verbose=False)
            self.i += 1
            self.log(name='val/validity', value=final_res.validity[0], on_epoch=True)
            self.log(name='val/uniqueness', value=final_res.uniqueness[0], on_epoch=True)
            self.log(name='val/novelty', value=final_res.novelty[0], on_epoch=True)
            self.log(name='val/mol_stable', value=final_res.mol_stable[0], on_epoch=True)
            self.log(name='val/atm_stable', value=final_res.atm_stable[0], on_epoch=True)
                  
    def reverse_sampling(
        self,
        num_graphs: int,
        empirical_distribution_num_nodes: Tensor,
        device: torch.device,
        verbose: bool = False,
        save_traj: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        
        batch_num_nodes = torch.multinomial(input=empirical_distribution_num_nodes,
                                            num_samples=num_graphs, replacement=True).to(device)
        batch_num_nodes = batch_num_nodes.clamp(min=1)
        batch = torch.arange(num_graphs, device=device).repeat_interleave(batch_num_nodes, dim=0)
        bs = int(batch.max()) + 1
        
        # initialiaze the 0-mean point cloud from N(0, I)
        pos = torch.randn(len(batch), 3,
                          device=device,
                          dtype=torch.get_default_dtype()
                          )
        pos = zero_mean(pos, batch=batch, dim_size=bs, dim=0)
        
        n = len(pos)
        
        # initialize the atom-types 
        atom_types = torch.multinomial(self.atoms_prior, num_samples=n, replacement=True)
        atom_types = F.one_hot(atom_types, self.num_atom_features).float()
        
        charge_types = torch.multinomial(self.charges_prior, num_samples=n, replacement=True)
        charge_types = F.one_hot(charge_types, self.num_charge_classes).float()
        
        #edge_index_local = radius_graph(x=pos,
        #                                r=self.hparams.cutoff_local,
        #                                batch=batch, 
        #                                max_num_neighbors=self.hparams.max_num_neighbors)
        
        edge_index_local = None 
        
        # edge types for FC graph 
        edge_index_global = torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]        
        nE = len(mask_i)
        edge_attr_triu = torch.multinomial(self.bonds_prior, num_samples=nE, replacement=True)
       
        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global = torch.stack([j, i], dim=0)
        edge_attr_global = torch.concat([edge_attr_triu, edge_attr_triu], dim=0)
        edge_index_global, edge_attr_global = sort_edge_index(edge_index=edge_index_global,
                                                              edge_attr=edge_attr_global, 
                                                              sort_by_row=False)
        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]    
        
        # some assert
        edge_attr_global_dense = torch.zeros(size=(n, n), device=device, dtype=torch.long)
        edge_attr_global_dense[edge_index_global[0], edge_index_global[1]] = edge_attr_global
        assert (edge_attr_global_dense - edge_attr_global_dense.T).sum().float() == 0.0

        edge_attr_global = F.one_hot(edge_attr_global, self.num_bond_classes).float()
        
        batch_edge_global = batch[edge_index_global[0]]     
                  
        pos_traj = []
        atom_type_traj = []
        charge_type_traj = []
        edge_type_traj = []
        
        chain = range(self.hparams.timesteps)
    
        iterator = tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        for timestep in iterator:
            t = torch.full(size=(bs, ), fill_value=timestep, dtype=torch.long, device=pos.device)
            temb = t / self.hparams.timesteps
            temb = temb.unsqueeze(dim=1)
            
            node_feats_in = torch.cat([atom_types, charge_types], dim=-1)
            out = self.model(
                x=node_feats_in,
                t=temb,
                pos=pos,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_global=edge_attr_global,
                batch=batch,
                batch_edge_global=batch_edge_global
            )
            
        
            rev_sigma = self.sde.reverse_posterior_sigma[t].unsqueeze(-1)
            sigmast = self.sde.sqrt_1m_alphas_cumprod[t].unsqueeze(-1)
            sigmas2t = sigmast.pow(2)
            alpha_bar_t = self.sde.alphas_cumprod[t].unsqueeze(-1)
            
            sqrt_alphas = self.sde.sqrt_alphas[t].unsqueeze(-1)
            sqrt_1m_alphas_cumprod_prev = torch.sqrt(1.0 - self.sde.alphas_cumprod_prev[t]).unsqueeze(-1)
            one_m_alphas_cumprod_prev = sqrt_1m_alphas_cumprod_prev.pow(2)
            sqrt_alphas_cumprod_prev = torch.sqrt(self.sde.alphas_cumprod_prev[t].unsqueeze(-1))
            one_m_alphas = self.sde.discrete_betas[t].unsqueeze(-1)
            
            coords_pred = out['coords_pred'].squeeze()
            atoms_pred, charges_pred = out['atoms_pred'].split([self.num_atom_features, self.num_charge_classes], dim=-1)
            atoms_pred = atoms_pred.softmax(dim=-1)
            # N x a_0
            edges_pred = out['bonds_pred'].softmax(dim=-1)
            # E x b_0
            charges_pred = charges_pred.softmax(dim=-1)
        
                
            # update fnc
            
            # positions/coords
            mean = sqrt_alphas[batch] * one_m_alphas_cumprod_prev[batch] * pos + \
                sqrt_alphas_cumprod_prev[batch] * one_m_alphas[batch] * coords_pred
            mean = (1.0 / sigmas2t[batch]) * mean
            std = rev_sigma[batch]
            noise = torch.randn_like(mean)
            noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)
            pos = mean + std * noise
    
            # atoms   
            rev_atoms = self.cat_atoms.reverse_posterior_for_every_x0(xt=atom_types, t=t[batch])
            # Eq. 4 in Austin et al. (2023) "Structured Denoising Diffusion Models in Discrete State-Spaces"
            # (N, a_0, a_t-1)
            unweighted_probs_atoms = (rev_atoms * atoms_pred.unsqueeze(-1)).sum(1)
            unweighted_probs_atoms[unweighted_probs_atoms.sum(dim=-1) == 0] = 1e-5
            # (N, a_t-1)
            probs_atoms = unweighted_probs_atoms / unweighted_probs_atoms.sum(-1, keepdims=True)
            atom_types =  F.one_hot(probs_atoms.multinomial(1,).squeeze(), num_classes=self.num_atom_features).float()

            # charges
            rev_charges = self.cat_charges.reverse_posterior_for_every_x0(xt=charge_types, t=t[batch])
            # Eq. 4 in Austin et al. (2023) "Structured Denoising Diffusion Models in Discrete State-Spaces"
            # (N, a_0, a_t-1)
            unweighted_probs_charges = (rev_charges * charges_pred.unsqueeze(-1)).sum(1)
            unweighted_probs_charges[unweighted_probs_charges.sum(dim=-1) == 0] = 1e-5
            # (N, a_t-1)
            probs_charges = unweighted_probs_charges / unweighted_probs_charges.sum(-1, keepdims=True)
            charge_types =  F.one_hot(probs_charges.multinomial(1,).squeeze(), num_classes=self.num_charge_classes).float()
            
            # edges
            edges_pred_triu = edges_pred[mask]
            # (E, b_0)
            edges_xt_triu = edge_attr_global[mask]
            # (E, b_t)
            rev_edges = self.cat_bonds.reverse_posterior_for_every_x0(xt=edges_xt_triu, t=t[batch[mask_i]])
            # (E, b_0, b_t-1)
            unweighted_probs_edges = (rev_edges * edges_pred_triu.unsqueeze(-1)).sum(1)
            unweighted_probs_edges[unweighted_probs_edges.sum(dim=-1) == 0] = 1e-5
            probs_edges = unweighted_probs_edges / unweighted_probs_edges.sum(-1, keepdims=True)
            edges_triu = probs_edges.multinomial(1,).squeeze()
            
            j, i = edge_index_global
            mask = j < i
            mask_i = i[mask]
            mask_j = j[mask]
            j = torch.concat([mask_j, mask_i])
            i = torch.concat([mask_i, mask_j])
            edge_index_global = torch.stack([j, i], dim=0)
            edge_attr_global = torch.concat([edges_triu, edges_triu], dim=0)
            edge_index_global, edge_attr_global = sort_edge_index(edge_index=edge_index_global,
                                                                edge_attr=edge_attr_global, 
                                                                sort_by_row=False)
            edge_attr_global = F.one_hot(edge_attr_global, num_classes=self.num_bond_classes).float()    
            
            
            if not self.hparams.fully_connected:
                edge_index_local = radius_graph(x=pos.detach(),
                                                r=self.hparams.cutoff_local,
                                                batch=batch, 
                                                max_num_neighbors=self.hparams.max_num_neighbors)
                
            #atom_integer = torch.argmax(atom_types, dim=-1)
            #bond_integer = torch.argmax(edge_attr_global, dim=-1)
            
            if save_traj:
                pos_traj.append(pos.detach())
                atom_type_traj.append(atom_types.detach())
                edge_type_traj.append(edge_attr_global.detach())
                charge_type_traj.append(charge_types.detach())
                
        return pos, atom_types, charge_types, edge_attr_global, edge_index_global, batch_num_nodes, [pos_traj, atom_type_traj, edge_type_traj]
    
    def coalesce_edges(self, edge_index, bond_edge_index, bond_edge_attr, n):
        edge_attr = torch.full(size=(edge_index.size(-1), ),
                               fill_value=0,
                               device=edge_index.device,
                               dtype=torch.long)
        edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
        edge_attr =  torch.cat([edge_attr, bond_edge_attr], dim=0)
        edge_index, edge_attr = coalesce(index=edge_index, value=edge_attr, m=n, n=n, op="max")
        return edge_index, edge_attr
    
    def forward(self, batch: Batch, t: Tensor):
        
        atom_types: Tensor = batch.x
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        n = batch.num_nodes
        bs = int(data_batch.max()) + 1
        
        bond_edge_index, bond_edge_attr = sort_edge_index(edge_index=bond_edge_index,
                                                          edge_attr=bond_edge_attr,
                                                          sort_by_row=False)
        
        if not hasattr(batch, "fc_edge_index"):
            edge_index_global = torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1)).int().fill_diagonal_(0)
            edge_index_global, _ = dense_to_sparse(edge_index_global)
            edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        else:
            edge_index_global = batch.fc_edge_index
        
        edge_index_global, edge_attr_global = self.coalesce_edges(edge_index=edge_index_global,
                                                                  bond_edge_index=bond_edge_index, 
                                                                  bond_edge_attr=bond_edge_attr,
                                                                  n=pos.size(0))
        
        edge_index_global, edge_attr_global = sort_edge_index(edge_index=edge_index_global,
                                                              edge_attr=edge_attr_global, 
                                                              sort_by_row=False)
        
        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]        
        edge_attr_triu = edge_attr_global[mask]
        edge_attr_triu_ohe = F.one_hot(edge_attr_triu, num_classes=self.num_bond_classes).float()
        t_edge = t[data_batch[mask_i]]
        probs = self.cat_bonds.marginal_prob(edge_attr_triu_ohe, t=t_edge)
        edges_t_given_0 = probs.multinomial(1,).squeeze()
        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global_perturbed = torch.stack([j, i], dim=0)
        edge_attr_global_perturbed = torch.concat([edges_t_given_0, edges_t_given_0], dim=0)
        edge_index_global_perturbed, edge_attr_global_perturbed = sort_edge_index(edge_index=edge_index_global_perturbed,
                                                                                  edge_attr=edge_attr_global_perturbed, 
                                                                                  sort_by_row=False)
        
        if not self.train:
            # do assertion when valdating
            edge_attr_global_dense_perturbed = torch.zeros(n, n, device=pos.device, dtype=torch.long)
            edge_attr_global_dense_perturbed[edge_index_global_perturbed[0], edge_index_global_perturbed[1]] = edge_attr_global_perturbed
            assert (edge_attr_global_dense_perturbed - edge_attr_global_dense_perturbed.T).float().mean().item() == 0.0
            assert torch.allclose(edge_index_global, edge_index_global_perturbed)
            
        edge_attr_global_perturbed = F.one_hot(edge_attr_global_perturbed, num_classes=self.num_bond_classes).float()
    

        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)
        
        # Coords: point cloud in R^3
        # sample noise for coords and recenter
        noise_coords_true = torch.randn_like(pos)
        noise_coords_true = zero_mean(noise_coords_true, batch=data_batch, dim_size=bs, dim=0)
        # center the true point cloud
        pos_centered = zero_mean(pos, data_batch, dim=0, dim_size=bs)
        # get signal and noise coefficients for coords
        mean_coords, std_coords = self.sde.marginal_prob(x=pos_centered, t=t[data_batch])
        # perturb coords
        pos_perturbed = mean_coords + std_coords * noise_coords_true
        
        # one-hot-encode
        if self.hparams.no_h:
            raise NotImplementedError
            node_feat -= 1 
            
        # one-hot-encode atom types
        atom_types = F.one_hot(atom_types.squeeze().long(), num_classes=self.num_atom_features).float()
        probs = self.cat_atoms.marginal_prob(atom_types.float(), t[data_batch])
        atom_types_perturbed = probs.multinomial(1,).squeeze()
        atom_types_perturbed = F.one_hot(atom_types_perturbed, num_classes=self.num_atom_features).float()
    
    
        # one-hot-encode charges
        # offset
        charges = charges + 2
        charges = F.one_hot(charges.squeeze().long(), num_classes=self.num_charge_classes).float()
        probs = self.cat_charges.marginal_prob(charges.float(), t[data_batch])
        charges_perturbed = probs.multinomial(1,).squeeze()
        charges_perturbed = F.one_hot(charges_perturbed, num_classes=self.num_charge_classes).float()
        
        batch_edge_global = data_batch[edge_index_global[0]]     
        
        #import pdb
        #print(ohes_perturbed.shape)
        #pdb.set_trace()
        atom_feats_in_perturbed = torch.cat([atom_types_perturbed, charges_perturbed], dim=-1)


        out = self.model(
            x=atom_feats_in_perturbed,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=None,
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global_perturbed,
            batch=data_batch,
            batch_edge_global=batch_edge_global,
        )
        
        out['coords_perturbed'] = pos_perturbed
        out['atoms_perturbed'] = atom_types_perturbed
        out['charges_perturbed'] = charges_perturbed
        out['bonds_perturbed'] = edge_attr_global_perturbed
        
        out['coords_true'] = pos_centered
        out['atoms_true'] = atom_types.argmax(dim=-1)
        out['bonds_true'] = edge_attr_global
        out['charges_true'] = charges.argmax(dim=-1)

        
        return out
    
    def loss_non_nans(self, loss: Tensor, modality: str) -> Tensor:
        m = loss.isnan()
        if torch.any(m):
            print(f"Recovered NaNs in {modality}. Selecting NoN-Nans")
        return loss[~m]
    
    def step_fnc(self, batch, batch_idx, stage: str):
        batch_size = int(batch.batch.max()) + 1
        t = torch.randint(low=0, high=self.hparams.timesteps,
                            size=(batch_size,), 
                            dtype=torch.long, device=batch.x.device)
        out_dict = self(batch=batch, t=t)
        
        
        #snr = self.sde.alphas_cumprod.pow(2) / (self.sde.sqrt_1m_alphas_cumprod.pow(2))
        #s = t - 1
        #s = torch.clamp(s, min=0)
        #w = snr[s] - snr[t]
        
        old = False
        
        coords_pred = out_dict['coords_pred']
        atoms_pred = out_dict['atoms_pred']
        atoms_pred, charges_pred = atoms_pred.split([self.num_atom_features, self.num_charge_classes], dim=-1)
        edges_pred = out_dict['bonds_pred']
        
        
        if old:
            w = 1 / batch_size
            coords_loss = torch.pow(
            coords_pred - out_dict["coords_true"], 2
            ).mean(-1)
        
            coords_loss = scatter_mean(
                coords_loss, index=batch.batch, dim=0, dim_size=batch_size
            )
            coords_loss = self.loss_non_nans(coords_loss, "coords")
            coords_loss *= w        
            coords_loss = torch.sum(coords_loss, dim=0)
            
            atoms_loss = F.cross_entropy(
                atoms_pred, out_dict["atoms_true"], reduction='none'
                )  
            atoms_loss = scatter_mean(
                atoms_loss, index=batch.batch, dim=0, dim_size=batch_size
            )
            atoms_loss = self.loss_non_nans(atoms_loss, "atoms")
            atoms_loss *= w
            atoms_loss = torch.sum(atoms_loss, dim=0)
            
            bonds_loss = F.cross_entropy(
                edges_pred, out_dict["bonds_true"], reduction='none'
            )
            
            bonds_loss = 0.5 * scatter_mean(
                bonds_loss, index=out_dict["edge_index"][1][1], dim=0, dim_size=out_dict["atoms_pred"].size(0)
            )
            bonds_loss = scatter_mean(
                bonds_loss, index=batch.batch, dim=0, dim_size=batch_size
            )
            bonds_loss = self.loss_non_nans(bonds_loss, "bonds")
            bonds_loss *= w
            bonds_loss = bonds_loss.sum(dim=0)
        else:
            coords_loss = self.mse_loss(coords_pred, out_dict["coords_true"])
            atoms_loss = self.ce_loss(atoms_pred, out_dict["atoms_true"])
            charges_loss = self.ce_loss(charges_pred, out_dict["charges_true"])
            bonds_loss = self.ce_loss(edges_pred, out_dict["bonds_true"])
        
       
        loss = 3.0 * coords_loss +  0.4 * atoms_loss +  2.0 * bonds_loss + 1.0 * charges_loss

        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=False,
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )

        self.log(
            f"{stage}/coords_loss",
            coords_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage=='train'),
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )

        self.log(
            f"{stage}/atoms_loss",
            atoms_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage=='train'),
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )
        
        self.log(
            f"{stage}/charges_loss",
            atoms_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage=='train'),
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )
        
        self.log(
            f"{stage}/bonds_loss",
            bonds_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage=='train'),
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )
      
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams["lr"], amsgrad=True, weight_decay=1e-12)
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    optimizer=optimizer,
        #    patience=self.hparams["lr_patience"],
        #    cooldown=self.hparams["lr_cooldown"],
        #    factor=self.hparams["lr_factor"],
        #)
        #scheduler = {
        #    "scheduler": lr_scheduler,
        #    "interval": "epoch",
        #    "frequency": self.hparams["lr_frequency"],
        #    "monitor": "val/loss",
        #    "strict": False,
        #}
        return [optimizer]# , [scheduler]

    

if __name__ == "__main__":
    
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

    from experiments.geom.hparams_coordsatomsbonds import add_arguments
    from experiments.geom.geom_dataset import GeomDataModule

    parser = ArgumentParser()
    parser = add_arguments(parser)
    hparams = parser.parse_args()
    
    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    if not os.path.isdir(hparams.save_dir + f"/run{hparams.id}/"):
        print("Creating directory")
        os.mkdir(hparams.save_dir + f"/run{hparams.id}/")
    print(f"Starting Run {hparams.id}")
    # ema_callback = ExponentialMovingAverage(decay=hparams.ema_decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.save_dir + f"/run{hparams.id}/",
        save_top_k=1,
        monitor="val/coords_loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(
        hparams.save_dir + f"/run{hparams.id}/", default_hp_metric=False
    )

    dataset_info = get_dataset_info("drugs", remove_h=False)
  
    print(f"Loading {hparams.dataset} Datamodule.")
   
    print("Using MIDI GEOM")
    if hparams.no_h:
        raise NotImplementedError
        exit()
        root = '/home/let55/workspace/projects/e3moldiffusion/geom/data_noH' 
    else:
        root = '/home/let55/workspace/projects/e3moldiffusion/experiments/geom/data' # delta
        root = '/sharedhome/let55/projects/e3moldiffusion/experiments/geom/data' # aws
        root = '/hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/data' # alpha
    print(root)
    datamodule = GeomDataModule(root=root,
                                batch_size=hparams.batch_size,
                                num_workers=hparams.num_workers,
                                pin_memory=True,
                                persistent_workers=True,
                                with_hydrogen=not hparams.no_h
                                )
    datamodule.prepare_data()
    datamodule.setup("fit")
        
    atom_types_distribution = datamodule.train_dataset.statistics.atom_types
    bond_types_distribution = datamodule.train_dataset.statistics.bond_types
    charge_types_distribution = datamodule.train_dataset.statistics.charge_types
    if charge_types_distribution.ndim == 2:
            charge_types_distribution = charge_types_distribution.mean(0) / charge_types_distribution.mean(0).sum()
        
    print("Atom type distribution: ", atom_types_distribution)
    print("Bond type distribution: ", bond_types_distribution)
    print("Charge type distribution: ", charge_types_distribution)
    
    train_smiles = load_pickle(os.path.join(root, "processed", "train_smiles.pickle"))
    
    model = Trainer(hparams=hparams.__dict__,
                    dataset_info=dataset_info,
                    smiles_list=list(train_smiles),
                    distributions = {"atoms": atom_types_distribution.float(),
                                     "bonds": bond_types_distribution.float(), 
                                     "charges": charge_types_distribution.float()
                                     }
                    )

    strategy = "ddp" if  hparams.gpus > 1 else "auto"
    
    trainer = pl.Trainer(
        accelerator="gpu" if hparams.gpus else "cpu",
        devices=hparams.gpus if hparams.gpus else None,
        strategy=strategy,
        logger=tb_logger,
        enable_checkpointing=True,
        accumulate_grad_batches=hparams.accum_batch,
        val_check_interval=hparams.eval_freq,
        gradient_clip_val=hparams.grad_clip_val,
        callbacks=[
            # ema_callback,
            lr_logger,
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=5),
            ModelSummary(max_depth=2),
        ],
        precision=hparams.precision,
        num_sanity_val_steps=2,
        max_epochs=hparams.num_epochs,
        detect_anomaly=hparams.detect_anomaly,
    )

    pl.seed_everything(seed=0, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )

