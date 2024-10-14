import argparse
from pathlib import Path
from Bio import PDB
from Bio.PDB import PDBParser
import string
from copy import deepcopy

def combine_multiple_pdbs(pdb_structures):
    index = 0
    main_structure = deepcopy(pdb_structures[0])
    # Set chains in structures and move to first structure
    for x, structure in enumerate(pdb_structures):
        for model in structure:
            for chain in model:
                _chain = string.ascii_uppercase[index]
                chain.id = _chain
                index += 1
                # Don't move chains of first structure
                if x == 0: continue
                chain.detach_parent()
                main_structure[0].add(chain)
    return main_structure

def save_pdb_structure(main_structure, outfile):
    io = PDB.PDBIO()
    io.set_structure(main_structure)
    io.save(outfile)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str)
    args = parser.parse_args()
    datapath = args.datapath
    pdb_files = sorted(list(Path(datapath).glob("*.pdb")))
    pdb_files = [p for p in pdb_files if "merged" not in str(p)]
    sdf_file = list(Path(datapath).glob("*.sdf"))[0]
    outfile = datapath + "/merged.pdb"
    pdb_structures = [PDBParser(QUIET=True).get_structure("", pdbfile) for pdbfile in pdb_files]
    print(f"Merging {len(pdb_structures)} pdb structures into a single pdb file")
    merged_structure = combine_multiple_pdbs(pdb_structures)
    if save_pdb_structure(merged_structure, outfile):
        print(f"Saved merged pdb to {outfile}")