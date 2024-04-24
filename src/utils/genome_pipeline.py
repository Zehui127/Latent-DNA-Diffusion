import os
import sys
import numpy as np
import pandas as pd
import torch
from genome_extractor import GenomeExtractor
from onehot import onehot_encoder

# CLI ARGUMENTS
n = len(sys.argv)

# DEFINE EXTRACTOR
SPECIES = sys.argv[1] if n > 1 else "human"
extractor = GenomeExtractor(SPECIES)
species = extractor._GENOME_NAMES[SPECIES]

# DEFINE TARGET PATH
ROOT = os.path.dirname(os.path.abspath("__file__"))
DATA_PATH = os.path.join(ROOT, "species", species)
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

max_samples = 10000
interval = 2048
metadata = pd.DataFrame(columns=['chrom', 'start_index'])
for c in extractor.fasta.keys():
    print(c)
    chrom = str(extractor.fasta[c]).upper() # original has upper and lower case

    tensor = torch.tensor([]) # shape = (K , interval, 4)
    meta_li = []

    start = 0
    chrom_length = len(chrom)
    sequence_section = chrom # this gets shorter as we remove N's
    while start < chrom_length and tensor.shape[0] < max_samples and sequence_section != "":
        # finds the end of the "N"-free section
        end = sequence_section.find("N")
        if end == -1: # no N is in the sequence
            end = chrom_length
        else:
            end += start
        
        sequence_section = chrom[start:end]

        # extract (2048x4) tensors from the section
        for i in range(start, end-interval, interval): 
            if tensor.shape[0] >= max_samples:
                break
            encoded_sequence = onehot_encoder([chrom[i:i+2048]]).view(1, 2048, 4)
            tensor = torch.cat((tensor, encoded_sequence), dim=0)
            meta_li.append(i)
        
        print(tensor.shape[0], "/", max_samples)
            
        # find the start of the next "N"-free section
        start = end # looks at next set of sequence
        sequence_section = chrom[start:]
        with_n = len(sequence_section) # chrom_length-start

        sequence_section = sequence_section.lstrip("N")
        without_n = len(sequence_section)
        start += with_n - without_n

    tensor = tensor.view(-1, 2048, 4)
    torch.save(tensor, os.path.join(DATA_PATH, species + '_' + c + '.pt'))

    new_row = pd.DataFrame({'chrom': [c], 'start_index': [meta_li]})
    metadata = pd.concat([metadata, new_row], ignore_index=True)

metadata.to_csv(os.path.join(DATA_PATH, species + '_meta' + '.csv'))