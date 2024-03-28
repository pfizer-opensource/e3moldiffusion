import subprocess

# Path to the text file containing wget commands
file_path = "/scratch1/cremej01/data/zinc3d/ZINC-downloader-3D-sdf.gz-2.wget"

# Open the file in read mode
with open(file_path, "r") as file:
    # Iterate over each line in the file
    for line in file:
        # Execute the wget command using subprocess
        subprocess.run(line, shell=True)
