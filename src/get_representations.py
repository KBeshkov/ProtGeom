import torch
import matplotlib.pyplot as plt
import esm
import numpy as np
import sys
sys.path.append("../src/")
from Bio import PDB

from os import walk
import pickle
prot_dir = '../data/pdbs/'
subfolders =  next(walk(prot_dir))[1]
subfolders.sort()

model_names = ['esm2_t6_8M_UR50D','esm2_t12_35M_UR50D',
               'esm2_t30_150M_UR50D','esm2_t33_650M_UR50D']
models = [esm.pretrained.esm2_t6_8M_UR50D,esm.pretrained.esm2_t12_35M_UR50D,
          esm.pretrained.esm2_t30_150M_UR50D,esm.pretrained.esm2_t33_650M_UR50D]

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

for n,mod in enumerate(models):
    model_esm, alphabet = mod()
    model_esm.eval()
    batch_converter = alphabet.get_batch_converter()

    coords_space = []
    prot_labels = []
    count = 0
    coords_esm_space = [[] for i in range(model_esm.num_layers)]
    
    
    for sub in subfolders:
        filenames = next(walk(prot_dir+sub+'/'), (None, None, []))[2] 
        valid_filenames = []
        print(sub)
        for fname in filenames:
            if fname[-3:]=='cif':
                parser = PDB.MMCIFParser(QUIET=True)
                structure = parser.get_structure("protein", prot_dir+sub+'/'+fname)      
                protein = ''
                coords = []
                prot_ids = []
                try:
                    for residue in structure[0]['A']:
                        if 'CA' in residue:
                            ca_atom = residue['CA']
                            coords.append(ca_atom.coord)  
                            prot_ids.append(residue.id[1])
                            protein += d3to1[residue.resname]
                    coords = np.array(coords)
                    if len(coords)>0:
                        coords_space.append(coords)
                        valid_filenames.append(fname)
                        prot_labels.append(count)

                        for l in range(model_esm.num_layers+1):
                            _,_,batch_token_orig = batch_converter([('original_protein', protein)])
                            original = model_esm(batch_token_orig, repr_layers=[l])["representations"][l][0, 0 : len(protein)].detach().numpy()
                            coords_esm_space[l].append(original)
                except:
                    continue
        with open('../data/reps/coords_space'+model_names[n]+'_'+sub+'.pickle', 'wb') as f:
            pickle.dump(coords_space, f)
        with open('../data/reps/coords_esm_space'+model_names[n]+'_'+sub+'.pickle', 'wb') as f:
            pickle.dump(coords_esm_space, f)