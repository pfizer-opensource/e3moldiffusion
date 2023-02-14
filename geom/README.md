## Get Data

1) Download the  `rdkit_folder.tar.gz` from https://dataverse.harvard.edu/file.xhtml?fileId=4327252&version=4.0
2) Extract the `tar.gz`
3) Change the variables `PROCESS_PATH` and `PATH` in `data.py`.
4) Run the `data.py` script. Might take some time. Arguments are currently parsed using `click` - To reduce computation time and memory, set `max_conformers` to smaller. Currently set to 1000.

Note that this script is executed on the Pfizer HPC.
