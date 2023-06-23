import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from typing import List, Tuple

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
from experiments.utils.config_file import get_dataset_info
from experiments.utils.sampling import (Molecule,
                                        analyze_stability_for_molecules)

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
                 dataset_info: dict,
                 smiles_list: list
                 ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.include_charges = False
        self.num_atom_features = self.hparams.num_atom_types + int(self.include_charges)
        self.num_bond_classes = 5
        
        self.i = 0
        self.dataset_info = dataset_info
        
        empirical_num_nodes = self._get_empirical_num_nodes()
        self.register_buffer(name='empirical_num_nodes', tensor=empirical_num_nodes)
        
        self.smiles_list = smiles_list
        self.edge_scaling = 1.00
        self.node_scaling = 1.00
        
        self.relative_pos = True

        self.model = DenoisingEdgeNetwork(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            num_layers=hparams["num_layers"],
            use_norm=hparams["use_norm"],
            use_cross_product=hparams["use_cross_product"],
            num_atom_types=self.num_atom_features,
            num_bond_types=self.num_bond_classes,
            rbf_dim=hparams["num_rbf"],
            edge_dim=hparams['edim'],
            cutoff_local=hparams["cutoff_upper"],
            vector_aggr="mean",
            local_global_model=hparams["fully_connected_layer"],
            fully_connected=hparams["fully_connected"],
            recompute_edge_attributes=True,
            recompute_radius_graph=True
        )  
        self.sde = DiscreteDDPM(beta_min=hparams["beta_min"],
                                beta_max=hparams["beta_max"],
                                N=hparams["timesteps"],
                                scaled_reverse_posterior_sigma=True,
                                schedule="cosine",
                                enforce_zero_terminal_snr=False)
        
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

        atom_types_integer = torch.argmax(atom_types, dim=-1)
        atom_types_integer_split = atom_types_integer.detach().cpu().split(batch_num_nodes.cpu().tolist(), dim=0)
        
        return pos_splits, atom_types_split, atom_types_integer_split, edge_types, edge_index_global, batch_num_nodes, trajs    
    
    
    @torch.no_grad()
    def run_evaluation(self, step: int, dataset_info, ngraphs: int = 4000, bs: int = 500,
                       verbose: bool = False, inner_verbose=False):
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
            pos_splits, _, atom_types_integer_split, \
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
            
            for positions, atom_types, edges in zip(pos_splits,
                                                    atom_types_integer_split,
                                                    edge_attrs_splits):
                molecule = Molecule(atom_types=atom_types, positions=positions,
                                    dataset_info=dataset_info,
                                    charges=torch.zeros(0, device=positions.device),
                                    bond_types=edges
                                    )
                molecule_list.append(molecule)
            
        
        res = analyze_stability_for_molecules(molecule_list=molecule_list, 
                                              dataset_info=dataset_info,
                                              smiles_train=self.smiles_list,
                                              bonds_given=True
                                             )

        if verbose:
            print(f'Run time={datetime.now() - start}')
        total_res = {k: v for k, v in zip(['validity', 'uniqueness', 'novelty'], res[1][0])}
        total_res.update(res[0])
        print(total_res)
        total_res = pd.DataFrame.from_dict([total_res])        
        print(total_res)
        
        total_res['step'] = step
        total_res['epoch'] = self.current_epoch
        save_dir = os.path.join(self.hparams.save_dir, 'run0', 'evaluation.csv')
        with open(save_dir, 'a') as f:
            total_res.to_csv(f, header=False)
        return total_res
    
    
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
        
        # initialize the atom-types 
        atom_types = torch.randn(pos.size(0), self.num_atom_features, device=device)
        
        edge_index_local = radius_graph(x=pos,
                                        r=self.hparams.cutoff_upper,
                                        batch=batch, 
                                        max_num_neighbors=self.hparams.max_num_neighbors)
        
        edge_index_global = torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        # sample symmetric edge-attributes
        edge_attrs = torch.randn((edge_index_global.size(0),
                                  edge_index_global.size(1),
                                  self.num_bond_classes),
                                  device=device, 
                                  dtype=torch.get_default_dtype())
        # symmetrize
        edge_attrs = 0.5 * (edge_attrs + edge_attrs.permute(1, 0, 2))
        assert torch.norm(edge_attrs - edge_attrs.permute(1, 0, 2)).item() == 0.0
        # get COO format (2, E)
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        # select in PyG formt (E, self.hparams.num_bond_types)
        edge_attr_global = edge_attrs[edge_index_global[0, :], edge_index_global[1, :], :]
        batch_edge_global = batch[edge_index_global[0]]     
    
        pos_traj = []
        atom_type_traj = []
        edge_type_traj = []
        
        chain = range(self.hparams.timesteps)
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
            edges_pred = out['bonds_pred'].softmax(dim=-1)
          
            # update fnc
            
            # positions/coords
            mean = sqrt_alphas[batch] * one_m_alphas_cumprod_prev[batch] * out['coords_perturbed'] + \
                sqrt_alphas_cumprod_prev[batch] * one_m_alphas[batch] * coords_pred
            mean = (1.0 / sigmas2t[batch]) * mean
            std = rev_sigma[batch]
            noise = torch.randn_like(mean)
            noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)
            pos = mean + std * noise
            
            if torch.any(pos.isnan()):
                import pdb
                print(t)
                print(pos)
                pdb.set_trace()

            # atoms 
            mean = sqrt_alphas[batch] * one_m_alphas_cumprod_prev[batch] * out['atoms_perturbed'] + \
                sqrt_alphas_cumprod_prev[batch] * one_m_alphas[batch] * atoms_pred
            mean = (1.0 / sigmas2t[batch]) * mean
            std = rev_sigma[batch]
            noise = torch.randn_like(mean)
            atom_types = mean + std * noise
            
            
            # edges
            mean = sqrt_alphas[batch_edge_global] * one_m_alphas_cumprod_prev[batch_edge_global] * out['bonds_perturbed'] + \
                sqrt_alphas_cumprod_prev[batch_edge_global] * one_m_alphas[batch_edge_global] * edges_pred
            mean = (1.0 / sigmas2t[batch_edge_global]) * mean
            std = rev_sigma[batch_edge_global]
            noise_edges = torch.randn_like(edge_attrs)
            noise_edges = 0.5 * (noise_edges + noise_edges.permute(1, 0, 2))
            noise_edges = noise_edges[edge_index_global[0, :], edge_index_global[1, :], :]
            edge_attr_global = mean + std * noise_edges
            
                
            if not self.hparams.fully_connected:
                edge_index_local = radius_graph(x=pos.detach(),
                                                r=self.hparams.cutoff_upper,
                                                batch=batch, 
                                                max_num_neighbors=self.hparams.max_num_neighbors)
                
                 # include local (noisy) edge-attributes based on radius graph indices
                edge_attrs = torch.zeros_like(edge_attrs)
                edge_attrs[edge_index_global[0], edge_index_global[1], :] = edge_attr_global
                
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
        
        node_feat: Tensor = batch.z
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        batch_num_nodes = torch.bincount(data_batch)
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        n = batch.num_nodes
        bs = int(data_batch.max()) + 1
        
        bond_edge_index, bond_edge_attr = sort_edge_index(edge_index=bond_edge_index,
                                                          edge_attr=bond_edge_attr,
                                                          sort_by_row=False)

        valencies_true = scatter_add(bond_edge_attr, index=bond_edge_index[0], dim=0, dim_size=n)
                
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
        
        # create block diagonal matrix
        dense_edge = torch.zeros(n, n, device=pos.device, dtype=torch.long)
        # populate entries with integer features 
        dense_edge[edge_index_global[0, :], edge_index_global[1, :]] = edge_attr_global        
        dense_edge_ohe = F.one_hot(dense_edge.view(-1, 1),
                                   num_classes=BOND_FEATURE_DIMS + 1).view(n, n, -1).float()
        
        assert torch.norm(dense_edge_ohe - dense_edge_ohe.permute(1, 0, 2)).item() == 0.0
        # edge-scaling
        dense_edge_ohe = self.edge_scaling * dense_edge_ohe
        
        # create symmetric noise for edge-attributes
        noise_edges = torch.randn_like(dense_edge_ohe)
        noise_edges = 0.5 * (noise_edges + noise_edges.permute(1, 0, 2))
        assert torch.norm(noise_edges - noise_edges.permute(1, 0, 2)).item() == 0.0
        
        signal = self.sde.sqrt_alphas_cumprod[t]
        std = self.sde.sqrt_1m_alphas_cumprod[t]
        
        signal_b = signal[data_batch].unsqueeze(-1).unsqueeze(-1)
        std_b = std[data_batch].unsqueeze(-1).unsqueeze(-1)
        dense_edge_ohe_perturbed = dense_edge_ohe * signal_b + noise_edges * std_b
    
        # retrieve as edge-attributes in PyG Format 
        edge_attr_global_perturbed = dense_edge_ohe_perturbed[edge_index_global[0, :], edge_index_global[1, :], :]
        #edge_attr_global_noise = noise_edges[edge_index_global[0, :], edge_index_global[1, :], :]
    
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
        node_feat = F.one_hot(
            node_feat.squeeze().long(), num_classes=max(self.hparams["atom_types"]) + 1
        ).float()[:, self.hparams["atom_types"]]
        
        xohe = self.node_scaling * node_feat
        # sample noise for OHEs in {0, 1}^NUM_CLASSES
        noise_ohes_true = torch.randn_like(xohe)
        mean_ohes, std_ohes = self.sde.marginal_prob(x=xohe, t=t[data_batch])
        # perturb OHEs
        ohes_perturbed = mean_ohes + std_ohes * noise_ohes_true
        
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
        out['valencies_true'] = valencies_true


        out['edge_index'] = (edge_index_local, edge_index_global)
        
        return out, data_batch, batch_edge_global
    
    def step_fnc(self, batch, batch_idx, stage: str):
        batch_size = int(batch.batch.max()) + 1
        t = torch.randint(low=0, high=self.hparams.timesteps,
                            size=(batch_size,), 
                            dtype=torch.long, device=batch.x.device)
        out_dict, node_batch, edge_batch = self(batch=batch, t=t)
        
        
        #snr = self.sde.alphas_cumprod.pow(2) / (self.sde.sqrt_1m_alphas_cumprod.pow(2))
        #s = t - 1
        #s = torch.clamp(s, min=0)
        #w = snr[s] - snr[t]
        w = 1.0 / batch_size
    
        sigmast = self.sde.sqrt_1m_alphas_cumprod[t]
        sigmas2t = sigmast.pow(2)
        alpha_bar_t = self.sde.alphas_cumprod[t]
        sqrt_alpha_bar_t = alpha_bar_t.sqrt()

        coords_pred = out_dict['coords_pred']
        atoms_pred = out_dict['atoms_pred']
        edges_pred = out_dict['bonds_pred']
        valencies_pred = out_dict['valencies_pred']       


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
        
        valencies_loss = F.cross_entropy(
            valencies_pred, out_dict["valencies_true"], reduction='none'
        )
        valencies_loss = scatter_mean(
            valencies_loss, index=batch.batch, dim=0, dim_size=batch_size
        )
        valencies_loss *= w
        valencies_loss = torch.sum(valencies_loss, dim=0)
        
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
            
        loss = 3.0 * coords_loss +  1.0 * atoms_loss +  2.0 * bonds_loss + 1.0 * rel_pos_loss + 1.0 * valencies_loss

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
        
        self.log(
            f"{stage}/valencies_loss",
            valencies_loss,
            on_step=True,
            batch_size=batch_size,
            sync_dist=self.hparams.gpus > 1 and stage == "val"
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
    
    import sys
    file_dir = os.path.dirname(__file__)
    sys.path.append(file_dir)
    
    from data import QM9DataModule
    from hparams_coordsatomsbonds import add_arguments

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
    
    
    model = Trainer(
        hparams=hparams.__dict__,
        dataset_info=dataset_info,
        smiles_list=list(datamodule.dataset.smiles)
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
    