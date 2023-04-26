from ase import Atoms
from xtb.ase.calculator import XTB
import argparse


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--output-path', default=None, type=str,
                        help='If set, saves the energy to a text file to the given path.')
    parser.add_argument('--data', type=dict, help='Input data as a dictionary containing atom types and coordinates')
    args = parser.parse_args()
    return args


def calculate_xtb_energy(data):
    atoms = Atoms(positions=data["pos"], numbers=data["atomic_numbers"])
    atoms.calc = XTB(method="GFN2-xTB")
    pot_e = atoms.get_potential_energy()

    return pot_e


if __name__ == "__main__":
    args = get_args()

    potential_energy = calculate_xtb_energy(args.data)
    if args.output_path is not None:
        f = open(args.output_path, "a")
        f.write(potential_energy)
        f.close()
