import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.decomposition import PCA
import esm
import numpy as np
import sys
sys.path.append("../src/")
from ShapeAnalysis import *
from Bio import PDB
from os import walk
import pickle
from tqdm import tqdm
prot_dir = '../data/pdbs/'
subfolders =  next(walk(prot_dir))[1]
subfolders.sort()

model_names = ['esm2_t33_650M_UR50D']#['esm2_t6_8M_UR50D','esm2_t12_35M_UR50D',
               #'esm2_t30_150M_UR50D']#,'esm2_t33_650M_UR50D']
models = [esm.pretrained.esm2_t33_650M_UR50D]#[esm.pretrained.esm2_t6_8M_UR50D, esm.pretrained.esm2_t12_35M_UR50D,
          #esm.pretrained.esm2_t30_150M_UR50D]#,esm.pretrained.esm2_t33_650M_UR50D]
effective_dims = [[] for _ in range(len(models))]


with open('../data/reps/coords_space.pickle','rb') as f:
    coords_space = pickle.load(f)
with open('../data/reps/prot_labels.pickle','rb') as f:
    prot_labels = pickle.load(f)


effective_dim_coords = effective_dim(coords_space)
def run_model(placeholder=0):
    for n, mod in enumerate(models):
        with open('../data/reps/coords_esm_space_'+model_names[n]+'_k.pickle','rb') as f:
            coords_esm_space = pickle.load(f)
        n_layers = mod()[0].num_layers
        for layer in tqdm(range(n_layers)):
            effective_dims[n].append(effective_dim(coords_esm_space[layer]))
 
    
        #    dmats[n][layer] = compute_dmat(coords_esm_space[layer], SA,subsamples=2,n_samples=10)
        #max_raidii[n].append([[np.nanmax(dmat_sub) for dmat_sub in dmat] for dmat in dmats[n]])
        
        fig, ax = plt.subplots(1,1,figsize=(3,3))
        #ax[0].plot(max_raidii[n][0],'o-',color='brown')
        #ax[0].axhline(y=max_radius_coords,color='k',linestyle='--')
        #ax[0].set_title('Max Radius')
        ax.plot(effective_dims[n],'o-',color='cyan')
        ax.axhline(y=effective_dim_coords,color='k',linestyle='--')
        ax.set_title('Effective Dimension PCA')
        ax.grid('on')
        fig.savefig('../figures/PCA_effdim_'+model_names[n]+'.png',dpi=300)
run_model(0)

