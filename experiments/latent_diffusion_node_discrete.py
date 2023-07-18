import logging
import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import (dense_to_sparse, dropout_node,
                                   sort_edge_index)
from tqdm import tqdm
from torch_scatter import scatter

from e3moldiffusion.latentdiffusion import (Encoder, Decoder, DenoisingLatentEdgeNetwork)
from e3moldiffusion.modules import GatedEquivBlock
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.losses import DiffusionLoss
from experiments.molecule_utils import Molecule
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.utils import (coalesce_edges, get_empirical_num_nodes,
                               get_list_of_edge_adjs, zero_mean)

logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.utilities.rank_zero").addHandler(logging.NullHandler())
logging.getLogger("pytorch_lightning.accelerators.cuda").addHandler(logging.NullHandler())

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]


def normalize_vector(v, eps: float = 1e-6):
    vnorm = torch.clamp(torch.pow(v, 2).sum(-1, keepdim=True), min=eps).sqrt()
    v = torch.div(v, vnorm)
    return v


def orthogonal_projection(v1, v2):
    dot_prod = torch.mul(v1, v2).sum(dim=-1, keepdim=True)
    v2 = v2 - dot_prod * v1
    return v2


def get_rotation_matrix_from_two_vector(v1, v2):
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)
    v2 = orthogonal_projection(v1, v2)
    v2 = normalize_vector(v2)
    v3 = torch.cross(v1, v2, dim=-1)
    rot = torch.stack((v1, v2, v3), dim=-1)
    return rot



class Trainer(pl.LightningModule):
    def __init__(self,
                 hparams: dict,
                 dataset_info: dict,
                 smiles_list: list,
                 dataset_statistics = None
                 ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.i = 0

        self.dataset_statistics = dataset_statistics

        atom_types_distribution = dataset_statistics.atom_types.float()
        bond_types_distribution = dataset_statistics.edge_types.float()
        charge_types_distribution = dataset_statistics.charges_marginals.float()

        self.register_buffer('atoms_prior', atom_types_distribution.clone())
        self.register_buffer('bonds_prior', bond_types_distribution.clone())
        self.register_buffer('charges_prior', charge_types_distribution.clone())
        
        self.hparams.num_atom_types = dataset_statistics.input_dims.X
        self.num_charge_classes = dataset_statistics.input_dims.C
        self.num_atom_types = self.hparams.num_atom_types
        self.num_atom_features = self.num_atom_types + self.num_charge_classes
        self.num_bond_classes = 5

        if hparams.get('no_h'):
            print("Training without hydrogen")
            self.hparams.num_atom_types -= 1
            
        self.smiles_list = smiles_list

        self.dataset_info = dataset_info
            
        empirical_num_nodes = get_empirical_num_nodes(dataset_info)
        self.register_buffer(name='empirical_num_nodes', tensor=empirical_num_nodes)

        self.encoder = Encoder(
            num_atom_features=self.num_atom_features,
            num_bond_types=self.num_bond_classes,
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            edge_dim=hparams['edim'],
            cutoff_local=hparams["cutoff_local"],
            num_layers=hparams["num_layers"],
            use_cross_product=False,
            vector_aggr="mean",
            atom_mapping=True,
            bond_mapping=True,
        )
        self.latent_lin = GatedEquivBlock(in_dims=(hparams["sdim_latent"], hparams["vdim_latent"]),
                                          out_dims=(hparams["latent_dim"], 2))  
        self.decoder = Decoder(
            num_atom_features=self.num_atom_features,
            num_bond_types=self.num_bond_classes,
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            edge_dim=hparams['edim'],
            num_layers=hparams["num_layers"],
            latent_dim=hparams["latent_dim"],
            use_cross_product=False,
            recompute_edge_attributes=True,
            vector_aggr="mean"
        )
        
        self.denoiser = DenoisingLatentEdgeNetwork(
            num_atom_features=self.num_atom_features,
            num_bond_types=self.num_bond_classes,
            in_dim=hparams["sdim"],
            edge_dim=hparams["edim"],
            num_layers=hparams["num_layers"],
            latent_dim=hparams["latent_dim"],
            atom_mapping=True,
            bond_mapping=True
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
        
        self.decoding_loss = DiffusionLoss(modalities=["coords", "atoms", "charges", "bonds"])
        self.diffusion_loss = DiffusionLoss(modalities=["latents", "atoms", "charges", "bonds"])


    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")
    
    def on_validation_epoch_end(self):
        
        if (self.current_epoch + 1) % self.hparams.test_interval == 0:  
            if self.local_rank == 0:
                print(f"Running evaluation in epoch {self.current_epoch + 1}")      
            final_res = self.run_evaluation(step=self.i, dataset_info=self.dataset_info,
                                            ngraphs=1000, bs=self.hparams.inference_batch_size,
                                            verbose=True, inner_verbose=False)
            self.i += 1
            self.log(name='val/validity', value=final_res.validity[0], on_epoch=True, sync_dist=True)
            self.log(name='val/uniqueness', value=final_res.uniqueness[0], on_epoch=True, sync_dist=True)
            self.log(name='val/novelty', value=final_res.novelty[0], on_epoch=True, sync_dist=True)
            self.log(name='val/mol_stable', value=final_res.mol_stable[0], on_epoch=True, sync_dist=True)
            self.log(name='val/atm_stable', value=final_res.atm_stable[0], on_epoch=True, sync_dist=True)
               
        
    def step_fnc(self, batch, batch_idx, stage: str):
        batch_size = int(batch.batch.max()) + 1
        t = torch.randint(low=1, high=self.hparams.timesteps + 1,
                            size=(batch_size,), 
                            dtype=torch.long, device=batch.x.device)
        out_dict = self(batch=batch, t=t)
        
       
        decoding_loss = self.decoding_loss(true_data=out_dict["true"],
                                           pred_data=out_dict["pred"]["decoder"],
                                           batch=batch.batch,
                                           bond_aggregation_index=out_dict["bond_aggregation_index"],
                                           weights=None
                                           )
        # encoder-decoder
        decoding_loss_aggr = 1.0 * decoding_loss["coords"] + 1.0 * decoding_loss["atoms"] +  1.0 * decoding_loss["bonds"] + 1.0 * decoding_loss["charges"]
        
        
        # diffusion
        diffusion_loss = self.diffusion_loss(true_data=out_dict["true"],
                                           pred_data=out_dict["pred"]["denoiser"],
                                           batch=batch.batch,
                                           bond_aggregation_index=out_dict["bond_aggregation_index"],
                                           weights=None
                                           )
        diffusion_loss_aggr = 1.0 * diffusion_loss["coords"] + 1.0 * diffusion_loss["atoms"] +  1.0 * diffusion_loss["bonds"] + 1.0 * diffusion_loss["charges"]
        
        final_loss = 0.5 * (decoding_loss_aggr * diffusion_loss_aggr)
        
        if torch.any(final_loss.isnan()):
            final_loss = final_loss[~final_loss.isnan()]
            print(f"Detected NaNs. Terminating training at epoch {self.current_epoch}")
            exit()
            
        self._log(final_loss, decoding_loss, diffusion_loss, batch_size, stage)

        return final_loss
    
    def forward(self, batch: Batch, t: Tensor):
        
        atom_types: Tensor = batch.x
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        n = batch.num_nodes
        bs = int(data_batch.max()) + 1
        
        pos_centered = pos - scatter(pos, index=batch, dim=0, reduce="mean")[batch]


        bond_edge_index, bond_edge_attr = sort_edge_index(edge_index=bond_edge_index,
                                                          edge_attr=bond_edge_attr,
                                                          sort_by_row=False)
        
        
        if not hasattr(batch, "fc_edge_index"):
            edge_index_global = torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1)).int().fill_diagonal_(0)
            edge_index_global, _ = dense_to_sparse(edge_index_global)
            edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        else:
            edge_index_global = batch.fc_edge_index
        
        
        edge_index_global, edge_attr_global = coalesce_edges(edge_index=edge_index_global,
                                                                  bond_edge_index=bond_edge_index, 
                                                                  bond_edge_attr=bond_edge_attr,
                                                                  n=pos.size(0))
        
        edge_index_global, edge_attr_global = sort_edge_index(edge_index=edge_index_global,
                                                              edge_attr=edge_attr_global, 
                                                              sort_by_row=False)
        
        # OHE the atom-types as well as charges
        atom_types = F.one_hot(atom_types.squeeze().long(), num_classes=self.num_atom_types).float()
        charges = self.dataset_statistics.one_hot_charges(charges)

        atom_features_in = torch.cat([atom_types, charges], dim=-1)
        edge_attr_global_ohe = F.one_hot(edge_attr_global, num_classes=self.num_bond_classes).float()

        # Encoder: mapping to latent space Z still on node-level. Z is only SE(3) invariant
        latent_out = self.encoder(x=atom_features_in,
                                  pos=pos_centered,
                                  edge_index_local=edge_index_global,
                                  edge_attr_local=edge_attr_global_ohe,
                                  batch=data_batch)
        latent_out, vector_out = self.latent_lin(x=(latent_out["s"], latent_out["v"]))
        vector_out = scatter(vector_out, index=batch, dim=0, reduce='add', dim_size=int(data_batch.max()) + 1)
        rot = get_rotation_matrix_from_two_vector(vector_out[:, :, 0], vector_out[:, :, 1])
        
        # Decoder: mapping from latent Z as well as Topology through Atom Features (A) and Edge-features (E) to Coordinate space as well graph space (A, E).
        decoder_out = self.decoder(
             x=atom_features_in,
             z=latent_out,
             rot=rot,
             edge_index_global=edge_index_global,
             edge_attr_global=edge_attr_global_ohe,
             batch=batch
        )
        # essentially the decoder needs to learn to map a latent point cloud to the ambient coordinate space


        # Denoising latent model that learns latent space Z as well as graph space (A, E)
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
        
        # latent space
        # sample noise for latents
        noise_latent_true = torch.randn_like(latent_out)
        # get signal and noise coefficients for latents
        mean_latent, std_latent = self.sde.marginal_prob(x=noise_latent_true, t=t[data_batch])
        # perturb latents
        latent_perturbed = mean_latent + std_latent * noise_latent_true
        
        # one-hot-encode
        if self.hparams.no_h:
            raise NotImplementedError
            node_feat -= 1 
            
        # perturb atom-types
        probs = self.cat_atoms.marginal_prob(atom_types.float(), t[data_batch])
        atom_types_perturbed = probs.multinomial(1,).squeeze()
        atom_types_perturbed = F.one_hot(atom_types_perturbed, num_classes=self.num_atom_types).float()
    
        # perturb charges
        probs = self.cat_charges.marginal_prob(charges.float(), t[data_batch])
        charges_perturbed = probs.multinomial(1,).squeeze()
        charges_perturbed = F.one_hot(charges_perturbed, num_classes=self.num_charge_classes).float()
        
        batch_edge_global = data_batch[edge_index_global[0]]     
        
        atom_feats_in_perturbed = torch.cat([atom_types_perturbed, charges_perturbed], dim=-1)

        denoising_out = self.denoiser(
            x=atom_feats_in_perturbed,
            z=latent_perturbed,
            t=temb,
            edge_index=edge_index_global,
            edge_attr=edge_attr_global_perturbed,
            batch=data_batch    ,
            batch_edge_global=batch_edge_global    
            )
        
        true_data = {"coords": pos_centered, 
                     "atoms": atom_types.argmax(dim=-1),
                     "bonds": edge_attr_global_ohe.argmax(dim=-1),
                     "charges": charges.argmax(dim=-1),
                     "latents": latent_out
                     }
        
        atoms_pred_decoder = decoder_out["atoms_pred"]
        atoms_pred_decoder, charges_pred_decoder = atoms_pred_decoder.split([self.num_atom_types, self.num_charge_classes], dim=-1)

        
        atoms_pred_denoising = denoising_out["atoms_pred"]
        atoms_pred_denoising, charges_pred_denoising = atoms_pred_denoising.split([self.num_atom_types, self.num_charge_classes], dim=-1)
        
        
        pred_data = {
            "decoder": {
                "atoms": atoms_pred_decoder,
                "charges": charges_pred_decoder,
                "coords": decoder_out["coords_pred"],
                "bonds": decoder_out["bonds_pred"]
            },
            "denoiser": {
                "atoms": atoms_pred_denoising,
                "charges": charges_pred_denoising,
                "latents": denoising_out["latent_pred"],
                "bonds": denoising_out["bonds_pred"]
            },
            "vectors": vector_out,
            "bond_aggregation_index": edge_index_global[1]
        }
        
        return {"true": true_data, "pred": pred_data}
    

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

        pos_splits = pos.detach().split(batch_num_nodes.cpu().tolist(), dim=0)
        
        charge_types_integer = torch.argmax(charge_types, dim=-1)
        # offset back
        charge_types_integer = charge_types_integer - self.dataset_statistics.charge_offset
        charge_types_integer_split = charge_types_integer.detach().split(batch_num_nodes.cpu().tolist(), dim=0)
        atom_types_integer = torch.argmax(atom_types, dim=-1)
        if self.hparams.no_h:
            raise NotImplementedError # remove in future or implement
            atom_types_integer += 1
            
        atom_types_integer_split = atom_types_integer.detach().split(batch_num_nodes.cpu().tolist(), dim=0)
        
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
        start = datetime.now()
        if verbose:
            if self.local_rank == 0:
                print(f"Creating {ngraphs} graphs in {l} batches")
        for _, num_graphs in enumerate(l):
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
            edge_attrs_splits = get_list_of_edge_adjs(edge_attrs_dense, batch_num_nodes)
            
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
                                                                                            local_rank=self.local_rank,
                                                                                            device=self.device
                                                                                            )

        run_time = datetime.now() - start
        if verbose:
            if self.local_rank == 0:
                print(f'Run time={run_time}')
        total_res = dict(stability_dict)
        total_res.update(validity_dict)
        if self.local_rank == 0:
            print(total_res)
        total_res = pd.DataFrame.from_dict([total_res])   
        if self.local_rank == 0:     
            print(total_res)
        total_res['step'] = str(step)
        total_res['epoch'] = str(self.current_epoch)
        total_res['run_time'] = str(run_time)
        total_res['ngraphs'] = str(ngraphs)
        try:
            if save_dir is None:
                save_dir = os.path.join(self.hparams.save_dir, 'run' + str(self.hparams.id), 'evaluation.csv')
                print(f"Saving evaluation csv file to {save_dir}")
            else:
                save_dir = os.path.join(save_dir, 'evaluation.csv')
            with open(save_dir, 'a') as f:
                total_res.to_csv(f, header=True)
        except Exception as e:
            print(e)
            pass
        return total_res
           
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
        
        # sample prior
        z = torch.randn(bs, self.hparams.latent_dim, device=device)
        
        # initialiaze the 0-mean point cloud from N(0, I)
        pos = torch.randn(len(batch), 3,
                          device=device,
                          dtype=torch.get_default_dtype()
                          )
        pos = zero_mean(pos, batch=batch, dim_size=bs, dim=0)
        
        n = len(pos)
        
        # initialize the atom-types 
        atom_types = torch.multinomial(self.atoms_prior, num_samples=n, replacement=True)
        atom_types = F.one_hot(atom_types, self.num_atom_types).float()
        
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
                z=z,
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
            atoms_pred, charges_pred = out['atoms_pred'].split([self.num_atom_types, self.num_charge_classes], dim=-1)
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
            atom_types =  F.one_hot(probs_atoms.multinomial(1,).squeeze(), num_classes=self.num_atom_types).float()

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
                
        return pos, atom_types, charge_types, edge_attr_global, edge_index_global, batch_num_nodes, [pos_traj, atom_type_traj, charge_type_traj, edge_type_traj]
    
    def _log(self, loss, decoder_loss: dict, diffusion_loss: dict, batch_size, stage):
        
        # total
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=False,
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )

        # decoder
        self.log(
            f"{stage}/dec_coords_loss",
            decoder_loss["coords"],
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage=='train'),
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )

        self.log(
            f"{stage}/dec_atoms_loss",
            decoder_loss["atoms"],
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage=='train'),
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )
        
        self.log(
            f"{stage}/dec_charges_loss",
            decoder_loss["charges"],
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage=='train'),
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )
        
        self.log(
            f"{stage}/dec_bonds_loss",
            decoder_loss["bonds"],
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage=='train'),
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )
        
        # diffusion
        self.log(
            f"{stage}/diff_latent_loss",
            diffusion_loss["latents"],
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage=='train'),
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )

        self.log(
            f"{stage}/diff_atoms_loss",
            diffusion_loss["atoms"],
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage=='train'),
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )
        
        self.log(
            f"{stage}/diff_charges_loss",
            diffusion_loss["charges"],
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage=='train'),
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )
        
        self.log(
            f"{stage}/diff_bonds_loss",
            diffusion_loss["bonds"],
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage=='train'),
            sync_dist=self.hparams.gpus > 1 and stage == "val"
        )
        
          
    def configure_optimizers(self):
        all_params = list(self.encoder.parameters()) \
                    + list(self.latent_lin.parameters()) \
                    + list(self.decoder.parameters()) \
                    + list(self.denoiser.parameters())
                                
        optimizer = torch.optim.AdamW(all_params, lr=self.hparams["lr"], amsgrad=True, weight_decay=1e-12)
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