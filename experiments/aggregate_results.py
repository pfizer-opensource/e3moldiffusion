import argparse
import os
from collections import defaultdict
from glob import glob

import numpy as np

from experiments.data.utils import load_pickle, save_pickle


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Aggregating evaluation dictionaries')
    parser.add_argument('--files-dir', type=str, help='Which dataset')
    parser.add_argument("--docked", default=False, action="store_true")
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
    name = "sampled" if not args.docked else "docked"

    if not args.docked:
        statistics_dicts = sorted(
            glob(os.path.join(args.files_dir, "*_statistics_dict.pickle"))
        )
        statistics_dict = aggregate_dicts(statistics_dicts)
        if args.remove_single_dicts:
            remove_dicts(statistics_dicts)
    else:
        score_dicts = sorted(
            glob(os.path.join(args.files_dir, "*_qvina2_scores.pickle"))
        )
        score_dict = aggregate_dicts(score_dicts)
        if args.remove_single_dicts:
            remove_dicts(score_dicts)

    buster_dicts = sorted(
        glob(os.path.join(args.files_dir, f"*_posebusters_{name}.pickle"))
    )
    buster_dict = aggregate_dicts(buster_dicts)
    if args.remove_single_dicts:
        remove_dicts(buster_dicts)

    violin_dicts = sorted(
        glob(os.path.join(args.files_dir, f"*_violin_dict_{name}.pickle"))
    )
    violin_dict = aggregate_dicts(violin_dicts)
    if args.remove_single_dicts:
        remove_dicts(violin_dicts)

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
            os.path.join(args.files_dir, "qvina2_scores.pickle"),
            exist_ok=True,
        )
        scores_mean = [np.mean(r) for r in score_dict["scores"] if len(r) >= 1]
        mean_score = np.mean(scores_mean)
        std_score = np.std(scores_mean)
        print(f"Mean docking score: {mean_score}")
        print(f"Docking score standard deviation: {std_score}")

        mean_top10_score = np.mean(
            [np.mean(sorted(r)[:10]) for r in score_dict["scores"] if len(r) >= 1]
        )
        print(f"Top-10 mean score: {mean_top10_score}")

    save_pickle(
        buster_dict,
        os.path.join(args.files_dir, f"posebusters_{name}.pickle"),
        exist_ok=False,
    )
    save_pickle(
        violin_dict,
        os.path.join(args.files_dir, f"violin_dict_{name}.pickle"),
        exist_ok=False,
    )
    save_pickle(
        posecheck_dict,
        os.path.join(args.files_dir, f"posecheck_{name}.pickle"),
        exist_ok=False,
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
    num_samples = len(violin_dict[list(violin_dict.keys())[0]])
    print(
        f"Found {num_sdf_files} sdf files in {os.path.join(args.files_dir, f'{name}')}. Check if that matches with 'num_ligands_per_pocket' specified in sampling."
    )
    print(f"Found {num_samples} ligands overall.")
    print(f"All files saved at {args.files_dir}.")


if __name__ == "__main__":
    args = get_args()
    aggregate(args)
