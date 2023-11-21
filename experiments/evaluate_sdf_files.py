import os
import warnings
from argparse import ArgumentParser

import pandas as pd

from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.utils import sdfs_to_molecules

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

from experiments.hparams_eval import add_arguments

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_arguments(parser)
    hparams = parser.parse_args()

    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    print(f"Loading {hparams.dataset} Datamodule.")
    non_adaptive = True
    if hparams.dataset == "drugs":
        dataset = "drugs"
        if hparams.use_adaptive_loader:
            non_adaptive = False
            from experiments.data.geom.geom_dataset_adaptive import (
                GeomDataModule as DataModule,
            )
        else:
            from experiments.data.geom.geom_dataset_nonadaptive import (
                GeomDataModule as DataModule,
            )
    elif hparams.dataset == "qm9":
        dataset = "qm9"
        from experiments.data.qm9.qm9_dataset import QM9DataModule as DataModule

    elif hparams.dataset == "aqm":
        dataset = "aqm"
        from experiments.data.aqm.aqm_dataset_nonadaptive import (
            AQMDataModule as DataModule,
        )
    elif hparams.dataset == "aqm_qm7x":
        dataset = "aqm_qm7x"
        from experiments.data.aqm_qm7x.aqm_qm7x_dataset_nonadaptive import (
            AQMQM7XDataModule as DataModule,
        )
    elif hparams.dataset == "pubchem":
        dataset = "pubchem"  # take dataset infos from GEOM for simplicity
        if hparams.use_adaptive_loader:
            non_adaptive = False
            from experiments.data.pubchem.pubchem_dataset_adaptive import (
                PubChemDataModule as DataModule,
            )
        else:
            from experiments.data.pubchem.pubchem_dataset_nonadaptive import (
                PubChemDataModule as DataModule,
            )
    elif hparams.dataset == "crossdocked":
        dataset = "crossdocked"
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
    elif hparams.dataset == "geomqm":
        dataset = "geomqm"
        from experiments.data.geom.geom_dataset_adaptive_qm import (
            GeomQMDataModule as DataModule,
        )
    else:
        raise ValueError(f"Unknown dataset: {hparams.dataset}")

    datamodule = DataModule(hparams)

    from experiments.data.data_info import GeneralInfos as DataInfos

    if dataset == "aqm_qm7x":
        from experiments.data.aqm.aqm_dataset_nonadaptive import (
            AQMDataModule as DataModule,
        )

        datamodule_aqm = DataModule(hparams)
        dataset_info = DataInfos(datamodule_aqm, hparams)
        del datamodule_aqm
    else:
        dataset_info = DataInfos(datamodule, hparams)

    train_smiles = (
        (
            list(datamodule.train_dataset.smiles)
            if hparams.dataset != "pubchem"
            else None
        )
        if not hparams.select_train_subset
        else datamodule.train_smiles
    )

    print("Converting SDF files to Molecules...")
    molecules = sdfs_to_molecules(hparams.sdf_path)
    print("Done.\n")

    print("Start analyzing...")

    (
        stability_dict,
        validity_dict,
        statistics_dict,
        all_generated_smiles,
        stable_molecules,
        valid_molecules,
    ) = analyze_stability_for_molecules(
        molecule_list=molecules,
        dataset_info=dataset_info,
        smiles_train=train_smiles,
        local_rank=0,
        return_molecules=True,
        remove_hs=hparams.remove_hs,
        device="cpu",
    )
    print("Done!")

    total_res = dict(stability_dict)
    total_res.update(validity_dict)
    total_res.update(statistics_dict)

    print(total_res)

    total_res = pd.DataFrame.from_dict([total_res])
    with open(hparams.save_dir, "a") as f:
        total_res.to_csv(f, header=True)
