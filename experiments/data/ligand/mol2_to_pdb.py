import os
from glob import glob

# List of Mol2 files
path = "/scratch1/cremej01/data/kinodata/raw/mol2/pocket"
mol2_files = glob(os.path.join(path, "*.mol2"))

# Output directory for PDB files
output_dir = os.path.join(path, "pdb")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over the Mol2 files and convert them to PDB
for mol2_file in mol2_files:
    # Construct the input and output file paths
    input_path = mol2_file
    output_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(mol2_file))[0] + ".pdb"
    )

    # Convert the Mol2 file to PDB using OpenBabel
    os.system(f"obabel {input_path} -O {output_path}")
