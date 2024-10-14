import argparse
import itertools
import os
import pickle
from collections import defaultdict
from glob import glob

import numpy as np
from rdkit import Chem

from experiments.sampling.walters_filter import walters_filter


def write_sdf_file(sdf_path, molecules, extract_mol=False):
    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if extract_mol:
            if m.rdkit_mol is not None:
                w.write(m.rdkit_mol)
        else:
            if m is not None:
                w.write(m)
    w.close()


def save_pickle(array, path, exist_ok=True):
    if exist_ok:
        with open(path, "wb") as f:
            pickle.dump(array, f)
    else:
        if not os.path.exists(path):
            with open(path, "wb") as f:
                pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Aggregating evaluation dictionaries')
    parser.add_argument('--files-dir', type=str, help='Which dataset')
    parser.add_argument("--docked", default=False, action="store_true")
    parser.add_argument("--docking-mode", default="qvina2", type=str)
    parser.add_argument("--walters-filter", default=False, action="store_true")
    parser.add_argument("--remove-single-dicts", default=False, action="store_true")
    args = parser.parse_args()
    return args


def aggregate_dicts(dicts):
    total_dict = defaultdict(list)
    for dict in dicts:
        d = load_pickle(dict)
        for k, v in d.items():
            total_dict[k].extend(v)
    return total_dict


def remove_dicts(dicts):
    for dict in dicts:
        os.remove(dict)


def aggregate(args):

    if not args.docked:
        statistics_dicts = sorted(
            glob(os.path.join(args.files_dir, "*_statistics_dict.pickle"))
        )
        statistics_dict = aggregate_dicts(statistics_dicts)
        if args.remove_single_dicts:
            remove_dicts(statistics_dicts)
    else:
        score_dicts = sorted(
            glob(os.path.join(args.files_dir, f"*_{args.docking_mode}_scores.pickle"))
        )
        score_dict = aggregate_dicts(score_dicts)
        if args.remove_single_dicts:
            remove_dicts(score_dicts)

    if (
        args.docked and args.docking_mode != "qvina2"
    ):  # ugly workaround for backward compatibility
        name = args.docking_mode
    elif args.docked and args.docking_mode == "qvina2":
        name = "qvina2"  # "docked"
    else:
        name = "sampled"

    violin_dicts = sorted(
        glob(os.path.join(args.files_dir, f"*_violin_dict_{name}.pickle"))
    )

    violin_dict = aggregate_dicts(violin_dicts)
    if args.remove_single_dicts:
        remove_dicts(violin_dicts)

    buster_dicts = sorted(
        glob(os.path.join(args.files_dir, f"*_posebusters_{name}.pickle"))
    )
    buster_dict = aggregate_dicts(buster_dicts)
    if args.remove_single_dicts:
        remove_dicts(buster_dicts)

    posecheck_dicts = sorted(
        glob(os.path.join(args.files_dir, f"*_posecheck_{name}.pickle"))
    )
    posecheck_dict = aggregate_dicts(posecheck_dicts)
    if args.remove_single_dicts:
        remove_dicts(posecheck_dicts)

    if not args.docked:
        save_pickle(
            statistics_dict,
            os.path.join(args.files_dir, "statistics_dict.pickle"),
            exist_ok=False,
        )
        statistics_dict = {
            k: {"mean": np.mean(v), "std": np.std(v)}
            for k, v in statistics_dict.items()
        }
        print(f"Mean statistics across all sampled ligands: {statistics_dict}")
    else:
        save_pickle(
            score_dict,
            os.path.join(args.files_dir, f"{args.docking_mode}_scores.pickle"),
            exist_ok=True,
        )
        if args.docking_mode == "qvina2":
            scores_mean = [np.mean(r) for r in score_dict["scores"] if len(r) >= 1]
            mean_score = np.mean(scores_mean)
            std_score = np.std(scores_mean)
            mean_top10_score = np.mean(
                [np.mean(sorted(r)[:10]) for r in score_dict["scores"] if len(r) >= 1]
            )
            print("\n")
            print(f"Mean QVina2 score (re-docking): {mean_score}")
            print(f"Std. QVina2 score (re-docking): {std_score}")
            print(f"Top-10 QVina2 mean score  (re-docking): {mean_top10_score}")
            print("\n")

        else:
            scores_mean = [np.mean(r) for r in score_dict["scores"] if len(r) >= 1]
            scores_mean_min = [
                np.mean(r) for r in score_dict["vina_minimize"] if len(r) >= 1
            ]
            mean_score = np.mean(scores_mean)
            mean_score_min = np.mean(scores_mean_min)
            std_score = np.std(scores_mean)
            std_score_min = np.std(scores_mean_min)
            mean_top10_score = np.mean(
                [np.mean(sorted(r)[:10]) for r in score_dict["scores"] if len(r) >= 1]
            )
            mean_top10_score_min = np.mean(
                [
                    np.mean(sorted(r)[:10])
                    for r in score_dict["vina_minimize"]
                    if len(r) >= 1
                ]
            )
            print("\n")
            print(f"Mean Vina score: {mean_score}")
            print(f"Std. Vina score: {std_score}")
            print(f"Top-10 mean Vina score: {mean_top10_score}")
            print("\n")
            print(f"Mean Vina score (minimized): {mean_score_min}")
            print(f"Std. Vina score (minimized): {std_score_min}")
            print(f"Top-10 mean Vina score (minimized): {mean_top10_score_min}")
            print("\n")

            if args.docking_mode == "vina_dock":
                scores_mean = [
                    np.mean(r) for r in score_dict["vina_dock"] if len(r) >= 1
                ]
                mean_score = np.mean(scores_mean)
                std_score = np.std(scores_mean)
                mean_top10_score = np.mean(
                    [
                        np.mean(sorted(r)[:10])
                        for r in score_dict["vina_dock"]
                        if len(r) >= 1
                    ]
                )
                print("\n")
                print(f"Mean Vina score (re-docking): {mean_score}")
                print(f"Vina score standard deviation (re-docking): {std_score}")
                print(f"Top-10 Vina mean score (re-docking): {mean_top10_score}")
                print("\n")

    save_pickle(
        violin_dict,
        os.path.join(args.files_dir, f"violin_dict_{name}.pickle"),
        exist_ok=True,
    )
    save_pickle(
        buster_dict,
        os.path.join(args.files_dir, f"posebusters_{name}.pickle"),
        exist_ok=True,
    )
    save_pickle(
        posecheck_dict,
        os.path.join(args.files_dir, f"posecheck_{name}.pickle"),
        exist_ok=True,
    )

    buster_dict = {
        k: {"mean": np.mean(v), "std": np.std(v)} for k, v in buster_dict.items()
    }
    posecheck_dict = {
        k: {"mean": np.mean(v), "std": np.std(v)} for k, v in posecheck_dict.items()
    }

    print(f"Mean PoseBusters metrics across all sampled ligands: {buster_dict}")
    print(f"Mean PoseCheck metrics across all sampled ligands: {posecheck_dict}")

    num_sdf_files = len(glob(os.path.join(args.files_dir, f"{name}/*.sdf")))
    if len(violin_dict) > 0:
        num_samples = len(violin_dict[list(violin_dict.keys())[0]])
    else:
        num_samples = 0

    print(
        f"Found {num_sdf_files} sdf files in {os.path.join(args.files_dir, f'{name}')}. Check if that matches with 'num_ligands_per_pocket' specified in sampling."
    )
    if len(violin_dict) > 0:
        print(f"Found {num_samples} ligands overall.")
    else:
        print("Posecheck was omitted. Violin dict was not creating after sample array.")
    print(f"All files saved at {args.files_dir}.")

    if args.walters_filter:
        print("Running Pat Walters filtering...")
        processed = "sampled" if not args.docked else args.docking_mode
        sdf_path = os.path.join(args.files_dir, "all_sdfs.sdf")
        mols_path = os.path.join(args.files_dir, processed)

        sdfs = glob(os.path.join(mols_path, "*.sdf"))
        mols = [
            [mol for mol in Chem.SDMolSupplier(sdf, sanitize=False)] for sdf in sdfs
        ]
        mols = list(itertools.chain(*mols))
        write_sdf_file(sdf_path, mols)
        args = {
            "sdf_path": sdf_path,
            "out_path": args.files_dir,
            "docking_mode": args.docking_mode if args.docked else None,
        }
        args = dotdict(args)
        walters_filter(args)


if __name__ == "__main__":
    args = get_args()
    aggregate(args)
