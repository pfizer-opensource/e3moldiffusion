# SAScore TEST WITH POCKET-CONDITION
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from rdkit import Chem
from torch_geometric.data import Batch
from tqdm import tqdm

from experiments.data.utils import mol_to_torch_geometric
from experiments.utils import prepare_pocket


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


extract_qvina2_score = lambda x: float(
    re.findall(r"[-+]?(?:\d*\.*\d+)", x.GetProp("REMARK"))[0]
)


test_dir = Path("/scratch1/cremej01/data/crossdocked_noH_cutoff5_new/test")
test_files = list(test_dir.glob("[!.]*.sdf"))
test_docked_files = (
    "/scratch1/cremej01/data/crossdocked_noH_cutoff5_new/test/evaluation/docking/docked"
)
test_docked_sdfs = glob(os.path.join(test_docked_files, "*.sdf"))

# model_path = "/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_bonds5_cutoff5_joint-pred-docking/run0/last-v4.ckpt"
model_path = "/scratch1/e3moldiffusion/logs/crossdocked/from_scratch_sascore_modelling_and_docking_new/best_valid.ckpt"


# load hyperparameter
hparams = torch.load(model_path)["hyper_parameters"]
hparams["select_train_subset"] = False
hparams["diffusion_pretraining"] = False
hparams["num_charge_classes"] = 6
hparams = dotdict(hparams)

hparams.load_ckpt_from_pretrained = None
hparams.store_intermediate_coords = False
hparams.load_ckpt = None
hparams.gpus = 1

print(f"Loading {hparams.dataset} Datamodule.")
if hparams.use_adaptive_loader:
    print("Using adaptive dataloader")
    from experiments.data.ligand.ligand_dataset_adaptive import (
        LigandPocketDataModule as DataModule,
    )
else:
    print("Using non-adaptive dataloader")
    from experiments.data.ligand.ligand_dataset_nonadaptive import (
        LigandPocketDataModule as DataModule,
    )

datamodule = DataModule(hparams)

from experiments.data.data_info import GeneralInfos as DataInfos

dataset_info = DataInfos(datamodule, hparams)
histogram = os.path.join(hparams.dataset_root, "size_distribution.npy")
histogram = np.load(histogram).tolist()
train_smiles = list(datamodule.train_dataset.smiles)

prop_norm, prop_dist = None, None

from experiments.diffusion_discrete_pocket import Trainer

# from experiments.property_training_pocket import Trainer

torch.cuda.empty_cache()

# if you want bond_model_guidance, flag this here in the Trainer
device = "cuda"
model = Trainer.load_from_checkpoint(
    model_path,
    dataset_info=dataset_info,
    smiles_list=train_smiles,
    histogram=histogram,
    prop_norm=prop_norm,
    prop_dist=prop_dist,
    load_ckpt_from_pretrained=None,
    store_intermediate_coords=False,
    ligand_pocket_hidden_distance=None,
    ligand_pocket_distance_loss=None,
    load_ckpt=None,
    # atoms_continuous=False,
    # joint_property_prediction=False,
    # energy_model_guidance=True if use_energy_guidance else False,
    # ckpt_energy_model=ckpt_energy_model,
    run_evaluation=True,
    strict=False,
).to(device)
model = model.eval()


failed = []
losses = []
for sdf_file in tqdm(test_files):
    ligand_name = sdf_file.stem

    pdb_name, pocket_id, *suffix = ligand_name.split("_")
    pdb_file = Path(sdf_file.parent, f"{pdb_name}.pdb")
    txt_file = Path(sdf_file.parent, f"{ligand_name}.txt")

    with open(txt_file, "r") as f:
        resi_list = f.read().split()

    try:
        pdb_struct = PDBParser(QUIET=True).get_structure("", pdb_file)[0]
        if resi_list is not None:
            # define pocket with list of residues
            residues = [
                pdb_struct[x.split(":")[0]][(" ", int(x.split(":")[1]), " ")]
                for x in resi_list
            ]
    except:
        failed.append(pdb_file)
        continue

    pocket_data = prepare_pocket(
        residues,
        dataset_info.atom_encoder,
        no_H=True,
        repeats=1,
        device=device,
        ligand_sdf=None,
    )

    suppl = Chem.SDMolSupplier(str(sdf_file))
    mol = []
    for m in suppl:
        mol.append(m)
    assert len(mol) == 1
    mol = mol[0]
    ligand_data = mol_to_torch_geometric(
        mol,
        dataset_info.atom_encoder,
        smiles=Chem.MolToSmiles(mol),
        remove_hydrogens=True,
        cog_proj=False,  # only for processing the ligand-shape encode
    )
    ligand_data = Batch.from_data_list([ligand_data]).to("cuda")

    for name, tensor in ligand_data.to_dict().items():
        pocket_data.__setattr__(name, tensor)

    t = torch.zeros((len(ligand_data),)).to("cuda").long()

    # get labels:
    sdf_file_docked = os.path.join(test_docked_files, f"{sdf_file.stem}_out.sdf")
    if sdf_file_docked in test_docked_sdfs:
        try:
            mol_docked = Chem.SDMolSupplier(str(sdf_file_docked))[0]
        except:
            failed.append(sdf_file)
            continue
        label = torch.tensor([extract_qvina2_score(mol_docked)]).to("cuda")
        pocket_data.docking_scores = label
    else:
        continue

    with torch.no_grad():
        preds = model(pocket_data, t=t)

    losses.append(
        F.mse_loss(preds["property_pred"][1].squeeze(1), label, reduction="none").item()
    )


print(f"Mean loss: {np.mean(losses)}")
print(f"Std loss: {np.std(losses)}")
