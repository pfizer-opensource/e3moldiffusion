#!/bin/sh

sbatch scripts/generate_ligands.sl

sbatch scripts/docking.sl
