import argparse
import subprocess


def split_list(data, num_chunks):
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        chunk_end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start:chunk_end])
        start = chunk_end
    return chunks


def process_files(args):
    file = open(args.path, "r")
    lines = file.readlines()
    file.close()

    line_list = []
    for line in lines:
        line = line.rstrip()
        line_list.append(line)

    lists = split_list(line_list, args.num_cpus)[args.mp_index - 1]

    for line in lists:
        # Execute the wget command using subprocess
        subprocess.run(line, shell=True)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--mp-index', default=0, type=int)
    parser.add_argument("--path", default="/hpfs/userws/cremej01/projects/data/zinc3d/ZINC-downloader-3D-sdf.gz-3.wget", type=str)
    parser.add_argument("--num-cpus", default=32, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    process_files(args)
