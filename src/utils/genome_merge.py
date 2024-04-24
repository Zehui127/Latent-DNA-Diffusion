import os
import sys
import torch

# CLI ARGUMENTS
n = len(sys.argv)
    
# DEFINE SPECIES
SPECIES = sys.argv[1] if n > 1 else "human"
SPECIES_MAP = {"human":"hg38", "mouse":"mm10"}
species = SPECIES_MAP[SPECIES]

ROOT = os.path.dirname(os.path.abspath("__file__"))
DATA_PATH = os.path.join(ROOT, "species", species)

FNAME = species + ".pt"
TARGET_PATH = os.path.join(ROOT, "merged")
if not os.path.exists(TARGET_PATH):
    os.makedirs(TARGET_PATH)

li = [f for f in os.listdir(DATA_PATH) if f.endswith('.pt')]
print("=== Merging sequence data for species:", species, len(li))
result = torch.tensor([])
for f in li:
    print("<<<", f, "...")
    tmp = torch.load(os.path.join(DATA_PATH, f))
    result = torch.concat([result, tmp], axis=0)

target = os.path.join(TARGET_PATH, FNAME)
torch.save(result, target)
print(">>> Saved to", target)