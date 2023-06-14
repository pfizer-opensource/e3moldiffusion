import logging
import os
from argparse import ArgumentParser
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
import torch.nn.functional as F 

from pytorch_lightning.loggers import TensorBoardLogger
from e3moldiffusion.molfeat import atom_type_config, get_bond_feature_dims
from e3moldiffusion.sde import VPAncestralSamplingPredictor, DiscreteDDPM
from e3moldiffusion.coordsatomsbonds import ScoreModelSE3, ScoreModel

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor
from torch_geometric.nn import radius_graph
from torch_sparse import coalesce
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean, scatter_add
from tqdm import tqdm

from e3moldiffusion.categorical import CategoricalTerminalKernel, CategoricalUniformKernel
#New imported 
from datetime import datetime
from callbacks.ema import ExponentialMovingAverage
from config_file import get_dataset_info
from qm9.utils_sampling import Molecule, analyze_stability_for_molecules
import pandas as pd


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
                 hparams,
                 dataset_info: dict
                 ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.include_charges = False
        self.num_atom_features = self.hparams.num_atom_types + int(self.include_charges)
        self.num_bond_classes = 5
        
        self.i = 0
        self.dataset_info = dataset_info
        
        self.node_scaling, self.edge_scaling = 1.0, 1.0
        
        self.mask_atoms = 1
        self.mask_edges = 1
        
        
        empirical_num_nodes = self._get_empirical_num_nodes()
        self.register_buffer(name='empirical_num_nodes', tensor=empirical_num_nodes)
    
        self.relative_pos = False
        
        self.model = ScoreModelSE3(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            num_layers=hparams["num_layers"],
            use_norm=hparams["use_norm"],
            use_cross_product=hparams["use_cross_product"],
            num_atom_types=self.num_atom_features + self.mask_atoms,
            num_bond_types=self.num_bond_classes + self.mask_edges,
            rbf_dim=hparams["num_rbf"],
            edge_dim=hparams['edim'],
            cutoff_local=hparams["cutoff_upper"],
            vector_aggr="mean",
            local_global_model=hparams["fully_connected_layer"],
            fully_connected=hparams["fully_connected"],
        )
        self.sde = DiscreteDDPM(beta_min=hparams["beta_min"],
                                beta_max=hparams["beta_max"],
                                N=hparams["timesteps"],
                                scaled_reverse_posterior_sigma=True,
                                schedule="cosine",
                                enforce_zero_terminal_snr=False)
        
        if self.mask_atoms:
            self.cat_atoms = CategoricalTerminalKernel(num_classes=self.num_atom_features + self.mask_atoms,
                                            terminal_state=self.num_atom_features,
                                            timesteps=hparams["timesteps"],
                                            alphas=self.sde.alphas.clone(),
                                           )
        else:
            self.cat_atoms = CategoricalUniformKernel(num_classes=self.num_atom_features + self.mask_atoms,
                                            timesteps=hparams["timesteps"],
                                            alphas=self.sde.alphas.clone(),
                                           )
            
        if self.mask_edges:
            self.cat_edges = CategoricalTerminalKernel(num_classes=self.num_bond_classes + self.mask_edges,
                                            terminal_state=self.num_bond_classes,
                                            timesteps=hparams["timesteps"],
                                            alphas=self.sde.alphas.clone())
        else:
            self.cat_edges = CategoricalUniformKernel(num_classes=self.num_bond_classes + self.mask_edges,
                                            timesteps=hparams["timesteps"],
                                            alphas=self.sde.alphas.clone())
        
        
    def _get_empirical_num_nodes(self):
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
        
        pos, atom_types, edge_types,\
        edge_index_global, batch_num_nodes, trajs = self.reverse_sampling(num_graphs=num_graphs,
                                                                          device=device,
                                                                          empirical_distribution_num_nodes=empirical_distribution_num_nodes,
                                                                          verbose=verbose,
                                                                          save_traj=save_traj)

        pos_splits = pos.detach().cpu().split(batch_num_nodes.cpu().tolist(), dim=0)
        atom_types_split = atom_types.detach().cpu().split(batch_num_nodes.cpu().tolist(), dim=0)
        
        if self.mask_atoms:
            atom_types_integer = torch.argmax(atom_types[:, :-1], dim=-1)
        else:
            atom_types_integer = torch.argmax(atom_types, dim=-1)
        
        if self.mask_edges:
            edge_types = edge_types[:, :-1]
            
        atom_types_integer_split = atom_types_integer.detach().cpu().split(batch_num_nodes.cpu().tolist(), dim=0)
        
        return pos_splits, atom_types_split, atom_types_integer_split, edge_types, edge_index_global, batch_num_nodes, trajs    
    
    
    @torch.no_grad()
    def run_evaluation(self, step: int, dataset_info, ngraphs: int = 4000, bs: int = 500):
        b = ngraphs // bs
        l = [bs] * b
        if sum(l) != ngraphs:
            l.append(ngraphs - sum(l))
        assert sum(l) == ngraphs
        
        results = []
        for num_graphs in l:
            start = datetime.now()
            pos_splits, _, atom_types_integer_split, \
            edge_types, edge_index_global, batch_num_nodes, _ = self.generate_graphs(
                                                                                     num_graphs=num_graphs,
                                                                                     verbose=False,
                                                                                     device=self.empirical_num_nodes.device,
                                                                                     empirical_distribution_num_nodes=self.empirical_num_nodes)
            
            n = batch_num_nodes.sum().item()
            edge_attrs_dense = torch.zeros(size=(n, n , 5), dtype=edge_types.dtype, device=edge_types.device)
            edge_attrs_dense[edge_index_global[0, :], edge_index_global[1, :], :] = edge_types
            edge_attrs_dense = edge_attrs_dense.argmax(-1)
            edge_attrs_splits = self.get_list_of_edge_adjs(edge_attrs_dense, batch_num_nodes)
            
            molecule_list = []
            for positions, atom_types, edges in zip(pos_splits,
                                                        atom_types_integer_split,
                                                        edge_attrs_splits):
                molecule = Molecule(atom_types=atom_types, positions=positions,
                                    dataset_info=dataset_info,
                                    charges=torch.zeros(0, device=positions.device),
                                    bond_types=edges)
                molecule_list.append(molecule)
            
        
            res = analyze_stability_for_molecules(molecule_list=molecule_list, 
                                                dataset_info=dataset_info,
                                                smiles_train=[], bonds_given=True
                                                )
            print(f'Run time={datetime.now() - start}')
            total_res = {k: v for k, v in zip(['validity', 'uniqueness', 'novelty'], res[1][0])}
            total_res.update(res[0])
            print(total_res)
            total_res = pd.DataFrame.from_dict([total_res])
            results.append(total_res)
            tmp_df = pd.concat(results)
            print(tmp_df.describe())
            print()
        
        final_res = dict(tmp_df.aggregate("mean", axis="rows"))
        final_res['step'] = step
        final_res['epoch'] = self.current_epoch
        final_res = pd.DataFrame.from_dict([final_res])

        save_dir = os.path.join(model.hparams.log_dir, 'evaluation.csv')
        with open(save_dir, 'a') as f:
            final_res.to_csv(f, header=False)
            
        return final_res
        
    def validation_epoch_end(self, validation_step_outputs):
        
        if (self.current_epoch + 1) % self.hparams.test_interval == 0:        
            final_res = self.run_evaluation(step=self.i, dataset_info=self.dataset_info, ngraphs=1000, bs=self.hparams.batch_size)
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
        
        n = len(batch)
        
        # initialize the atom-types as mask tokens
        atom_types = torch.zeros(pos.size(0), self.num_atom_features + self.mask_atoms, device=device, dtype=torch.float32)
        if self.mask_atoms:
            atom_types[:, -1] = 1.0
        else:
            atom_types[:, 1] = 1.0 # carbon
        
        edge_index_local = radius_graph(x=pos,
                                        r=self.hparams.cutoff_upper,
                                        batch=batch, 
                                        max_num_neighbors=self.hparams.max_num_neighbors)
        
        # edge_dense_placeholder = torch.zeros(size=(n, n), device=device, dtype=torch.long)
        
        edge_index_global = torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        # initialize the edge-types as mask tokens 
        #edge_attr_global = torch.zeros(edge_index_global.size(1), self.num_bond_classes + 1, device=device, dtype=torch.float32)
        #edge_attr_global[:, -1] = 1.0
        
        edge_attr_global_dense = torch.zeros(size=(n, n), device=device, dtype=torch.long)
        
        edge_attr_global = torch.zeros(edge_index_global.size(1), self.num_bond_classes + self.mask_edges, device=device, dtype=torch.float32)
        if self.mask_edges:
            edge_attr_global[:, -1] = 1.0
        else:
            edge_attr_global[:, 0] = 1.0  # no bond
        
        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]
        #edge_attr_triu = edge_attr_global[mask]
        batch_edge_global = batch[edge_index_global[0]]     

        pos_traj = []
        atom_type_traj = []
        edge_type_traj = []
        
        chain = range(self.hparams.timesteps)
    
        if verbose:
            print(chain)
        iterator = tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        for timestep in iterator:
            t = torch.full(size=(bs, ), fill_value=timestep, dtype=torch.long, device=pos.device)
            temb = t / self.hparams.timesteps
            temb = temb.unsqueeze(dim=1)
            
            out = self.model(
                x=atom_types,
                t=temb,
                pos=pos,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_local=None,
                edge_attr_global=edge_attr_global,
                batch=batch,
                batch_edge_global=batch_edge_global
            )
             
            rev_sigma = self.sde.reverse_posterior_sigma[t].unsqueeze(-1)
            sigmast = self.sde.sqrt_1m_alphas_cumprod[t].unsqueeze(-1)
            sigmas2t = sigmast.pow(2)
            
            sqrt_alphas = self.sde.sqrt_alphas[t].unsqueeze(-1)
            sqrt_1m_alphas_cumprod_prev = torch.sqrt(1.0 - self.sde.alphas_cumprod_prev[t]).unsqueeze(-1)
            one_m_alphas_cumprod_prev = sqrt_1m_alphas_cumprod_prev.pow(2)
            sqrt_alphas_cumprod_prev = torch.sqrt(self.sde.alphas_cumprod_prev[t].unsqueeze(-1))
            one_m_alphas = self.sde.discrete_betas[t].unsqueeze(-1)
            
            coords_pred = out['coords_pred'].squeeze()
            atoms_pred = out['atoms_pred'].softmax(dim=-1)
            # N x a
            edges_pred = out['bonds_pred'].softmax(dim=-1)
            # E x b
              
            # update fnc
            
            # positions/coords
            mean = sqrt_alphas[batch] * one_m_alphas_cumprod_prev[batch] * out['coords_perturbed'] + \
                sqrt_alphas_cumprod_prev[batch] * one_m_alphas[batch] * coords_pred
            mean = (1.0 / sigmas2t[batch]) * mean
            std = rev_sigma[batch]
            noise = torch.randn_like(mean)
            noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)
            pos = mean + std * noise
            
            # atoms  
            rev_atoms = self.cat_atoms.reverse_posterior_for_every_x0(xt=atom_types, t=t[batch])
            # (N, a, a)
            probs_atoms = (rev_atoms * atoms_pred.unsqueeze(-1)).sum(1)
            #if not (abs(probs_atoms.sum(-1) - 1.0) < 1e-4).all():
            #    print(probs_atoms)
            atom_types =  F.one_hot(probs_atoms.multinomial(1,).squeeze(), num_classes=self.num_atom_features + self.mask_atoms).float()
 
            # edges
            edges_pred_triu = edges_pred[mask]
            #print(edges_pred_triu)
            edges_xt_triu = edge_attr_global[mask]
            rev_edges = self.cat_edges.reverse_posterior_for_every_x0(xt=edges_xt_triu, t=t[batch[mask_i]])
            # (E, b, b)
            probs_edges = (rev_edges * edges_pred_triu.unsqueeze(-1)).sum(1)
            #if not (abs(probs_edges.sum(-1) - 1.0) < 1e-4).all():
            #    print(probs_edges)  
            
            #import pdb; pdb.set_trace()
            
            edges_triu = probs_edges.multinomial(1,).squeeze()
            # create full edge tensors
            #j = torch.concat([mask_j, mask_i]) # already computed above
            #i = torch.concat([mask_i, mask_j]) # already computed above
            #edge_index_global_perturbed = torch.stack([j, i], dim=0)
            edge_attr_global_dense[mask_j, mask_i] = edges_triu
            edge_attr_global_dense = 0.5 * (edge_attr_global_dense + edge_attr_global_dense.T)
            edge_attr_global_dense = edge_attr_global_dense.long()
            edge_attr_global = edge_attr_global_dense[j, i]
            edge_attr_global = F.one_hot(edge_attr_global, num_classes=self.num_bond_classes + self.mask_edges).float()
            
            # edge_attr_global = torch.concat([edges_triu, edges_triu], dim=0)
            
            if not self.hparams.fully_connected:
                edge_index_local = radius_graph(x=pos.detach(),
                                                r=self.hparams.cutoff_upper,
                                                batch=batch, 
                                                max_num_neighbors=self.hparams.max_num_neighbors)
                
            #atom_integer = torch.argmax(atom_types, dim=-1)
            #bond_integer = torch.argmax(edge_attr_global, dim=-1)
            
            if save_traj:
                pos_traj.append(pos.detach())
                atom_type_traj.append(atom_types.detach())
                edge_type_traj.append(edge_attr_global.detach())
                
        return pos, atom_types, edge_attr_global, edge_index_global, batch_num_nodes, [pos_traj, atom_type_traj, edge_type_traj]
    
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
        
        node_feat: Tensor = batch.x
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        batch_num_nodes = torch.bincount(data_batch)
        bond_edge_index = batch.bond_index
        bond_edge_attr = batch.bond_attr
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
        
        
        #edge_global_dense_true = torch.zeros(n, n, device=pos.device, dtype=torch.long)
        #edge_global_dense_true[edge_index_global[0], edge_index_global[1]] = edge_attr_global
        
        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]
        edge_attr_triu = edge_attr_global[mask]
        
        if self.mask_edges:
            edge_attr_triu_ohe = F.one_hot(edge_attr_triu, num_classes=self.num_bond_classes + 1).float()
        else:
            edge_attr_triu_ohe = F.one_hot(edge_attr_triu, num_classes=self.num_bond_classes).float()

        t_edge = t[data_batch[mask_i]]
        probs = self.cat_edges.marginal_prob(edge_attr_triu_ohe, t=t_edge)
        edges_t_given_0 = probs.multinomial(1,).squeeze()
        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global_perturbed = torch.stack([j, i], dim=0)
        edge_attr_global_perturbed = torch.concat([edges_t_given_0, edges_t_given_0], dim=0)
        edge_index_global_perturbed, edge_attr_global_perturbed = sort_edge_index(edge_index=edge_index_global_perturbed,
                                                                                  edge_attr=edge_attr_global_perturbed, 
                                                                                  sort_by_row=False)
        
        edge_attr_global_dense_perturbed = torch.zeros(n, n, device=pos.device, dtype=torch.long)
        edge_attr_global_dense_perturbed[edge_index_global_perturbed[0], edge_index_global_perturbed[1]] = edge_attr_global_perturbed
        assert (edge_attr_global_dense_perturbed - edge_attr_global_dense_perturbed.T).float().mean().item() == 0.0
        
        if not torch.allclose(edge_index_global, edge_index_global_perturbed):
            import pdb
            pdb.set_trace()
        

        edge_attr_global_perturbed = F.one_hot(edge_attr_global_perturbed, num_classes=self.num_bond_classes + self.mask_edges).float()
             
        if not self.hparams.continuous:
            temb = t.float() / self.hparams.timesteps
            temb = temb.clamp(min=self.hparams.eps_min)
        else:
            temb = t
            
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
        node_feat = F.one_hot(node_feat.squeeze().long(), num_classes=self.num_atom_features + self.mask_edges).float()
        xohe = self.node_scaling * node_feat
        probs = self.cat_atoms.marginal_prob(xohe.float(), t[data_batch])
        ohes_perturbed = probs.multinomial(1,).squeeze()
        ohes_perturbed = F.one_hot(ohes_perturbed, num_classes=self.num_atom_features + self.mask_edges ).float()
        
        
        edge_index_local = radius_graph(x=pos_perturbed,
                                        r=self.hparams.cutoff_upper,
                                        batch=data_batch, 
                                        flow="source_to_target",
                                        max_num_neighbors=self.hparams.max_num_neighbors)
        
        batch_edge_global = data_batch[edge_index_global[0]]     
        
        out = self.model(
            x=ohes_perturbed,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=edge_index_local,
            edge_index_global=edge_index_global,
            edge_attr_local=None,
            edge_attr_global=edge_attr_global_perturbed,
            batch=data_batch,
            batch_edge_global=batch_edge_global,
        )
        
        if "coords_perturbed" not in out.keys():
            out['coords_perturbed'] = pos_perturbed
        if "atoms_perturbed" not in out.keys():
            out['atoms_perturbed'] = ohes_perturbed
        if "bonds_perturbed" not in out.keys():
            out['bonds_perturbed'] = edge_attr_global_perturbed
        
        out['coords_true'] = pos_centered
        out['atoms_true'] = node_feat.argmax(dim=-1)
        out['bonds_true'] = edge_attr_global
        
        out['edge_index'] = (edge_index_local, edge_index_global)
        
        return out, data_batch, batch_edge_global
    
    def step_fnc(self, batch, batch_idx, stage: str):
        batch_size = int(batch.batch.max()) + 1
        t = torch.randint(low=0, high=self.hparams.timesteps,
                            size=(batch_size,), 
                            dtype=torch.long, device=batch.x.device)
        out_dict, node_batch, edge_batch = self(batch=batch, t=t)
        
  
        w = 1.0 / batch_size
    
   
        coords_pred = out_dict['coords_pred']
        atoms_pred = out_dict['atoms_pred']
        edges_pred = out_dict['bonds_pred']
           
                
        coords_loss = torch.pow(
           coords_pred - out_dict["coords_true"], 2
        ).sum(-1)
        
        coords_loss = scatter_mean(
            coords_loss, index=batch.batch, dim=0, dim_size=batch_size
        )
        coords_loss *= w        
        coords_loss = torch.sum(coords_loss, dim=0)
        
        atoms_loss = F.cross_entropy(
            atoms_pred, out_dict["atoms_true"], reduction='none'
            )  
        atoms_loss = scatter_mean(
            atoms_loss, index=batch.batch, dim=0, dim_size=batch_size
        )
        atoms_loss *= w
        atoms_loss = torch.sum(atoms_loss, dim=0)


        bonds_loss = F.cross_entropy(
            edges_pred, out_dict["bonds_true"], reduction='none'
        )
         
        bonds_loss = 0.5 * scatter_mean(
            bonds_loss, index=out_dict["edge_index"][1][1], dim=0, dim_size=out_dict["coords_true"].size(0)
        )
        bonds_loss = scatter_mean(
            bonds_loss, index=batch.batch, dim=0, dim_size=batch_size
        )
        bonds_loss *= w
        bonds_loss = bonds_loss.sum(dim=0)
        
        if self.relative_pos:
            j, i = out_dict["edge_index"][1]
            # distance 
            # pji_true = torch.pow(out_dict["coords_true"][j] - out_dict["coords_true"][i], 2).sum(-1, keepdim=True).sqrt()
            # pji_pred = torch.pow(coords_pred[j] - coords_pred[i], 2).sum(-1, keepdim=True).sqrt()
            # pos
            pji_true = out_dict["coords_true"][j] - out_dict["coords_true"][i]
            pji_pred = coords_pred[j] - coords_pred[i]
            rel_pos_loss = (pji_true - pji_pred).pow(2).mean(-1)
            rel_pos_loss = 0.5 * scatter_mean(rel_pos_loss, index=i, dim=0, dim_size=out_dict["coords_true"].size(0))
            rel_pos_loss = scatter_mean(rel_pos_loss, index=batch.batch, dim=0, dim_size=out_dict["coords_true"].size(0))
            rel_pos_loss *= w
            rel_pos_loss = rel_pos_loss.sum(dim=0)
        else:
            rel_pos_loss = 0.0
            
        loss = 3.0 * coords_loss +  1.0 * atoms_loss +  2.0 * bonds_loss + 1.0 * rel_pos_loss


        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}/coords_loss",
            coords_loss,
            on_step=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}/atoms_loss",
            atoms_loss,
            on_step=True,
            batch_size=batch_size,
        )
        
        self.log(
            f"{stage}/bonds_loss",
            bonds_loss,
            on_step=True,
            batch_size=batch_size,
        )
        
        self.log(
            f"{stage}/rel_coords_loss",
            rel_pos_loss,
            on_step=True,
            batch_size=batch_size,
        )
        
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=self.hparams["lr_patience"],
            cooldown=self.hparams["lr_cooldown"],
            factor=self.hparams["lr_factor"],
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams["lr_frequency"],
            "monitor": "val/loss",
            "strict": False,
        }
        return [optimizer], [scheduler]

    

if __name__ == "__main__":
    from qm9.data import QM9DataModule
    from qm9.hparams_coordsatomsbonds import add_arguments

    parser = ArgumentParser()
    parser = add_arguments(parser)
    hparams = parser.parse_args()
    
    if not os.path.exists(hparams.log_dir):
        os.makedirs(hparams.log_dir)

    if not os.path.isdir(hparams.log_dir + f"/run{hparams.id}/"):
        print("Creating directory")
        os.mkdir(hparams.log_dir + f"/run{hparams.id}/")
    print(f"Starting Run {hparams.id}")
    
    ema_callback = ExponentialMovingAverage(decay=hparams.ema_decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.log_dir + f"/run{hparams.id}/",
        save_top_k=1,
        monitor="val/coords_loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(
        hparams.log_dir + f"/run{hparams.id}/", default_hp_metric=False
    )

    print(f"Loading {hparams.dataset} Datamodule.")
    datamodule = QM9DataModule(hparams)
    datamodule.prepare_data()
    datamodule.setup("fit")

    dataset_info = get_dataset_info(hparams.dataset, hparams.remove_hs)


    dataloader = datamodule.get_dataloader(datamodule.train_dataset, "val")
        
    model = Trainer(
        hparams=hparams.__dict__,
        dataset_info=dataset_info
    )

    strategy = (
        pl.strategies.DDPStrategy(find_unused_parameters=False)
        if hparams.gpus > 1
        else None
    )

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
            ema_callback,
            lr_logger,
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=5),
            ModelSummary(max_depth=2),
        ],
        precision=hparams.precision,
        num_sanity_val_steps=2,
        max_epochs=hparams.num_epochs,
        detect_anomaly=hparams.detect_anomaly,
        resume_from_checkpoint=None
        if hparams.load_model is None
        else hparams.load_model,
    )

    pl.seed_everything(seed=0, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        # ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )
    