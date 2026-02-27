import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colormaps
import esm
import numpy as np
import sys
sys.path.append("../src/")
from dynamic_rep_loading import load_representations
from Bio import PDB
from ShapeAnalysis import *
import multiprocessing
from os import walk
import pickle
from tqdm import tqdm
prot_dir = '../data/pdbs/'
subfolders =  next(walk(prot_dir))[1]
subfolders.sort()

n_res = [1000,1000,500,1500]
sp_order = [2,3,2,2]
#SA = ShapeAnalysis(res=n_res,spline_order=sp_order,id=0)
model_names = ['esm2_t12_35M_UR50D','esm2_t12_35M_UR50D','esm2_t12_35M_UR50D','esm2_t12_35M_UR50D']#,'esm2_t12_35M_UR50D',
               #'esm2_t30_150M_UR50D']#,'esm2_t33_650M_UR50D']
models = [esm.pretrained.esm2_t12_35M_UR50D,esm.pretrained.esm2_t12_35M_UR50D,esm.pretrained.esm2_t12_35M_UR50D,esm.pretrained.esm2_t12_35M_UR50D]#, esm.pretrained.esm2_t12_35M_UR50D,
          #esm.pretrained.esm2_t30_150M_UR50D]#,esm.pretrained.esm2_t33_650M_UR50D]
dmats = [[] for _ in range(len(models))]
frechet_radii = [[] for _ in range(len(models))]
max_raidii = [[] for _ in range(len(models))]
effective_dims = [[] for _ in range(len(models))]
colormap = colormaps['tab10']


with open('../data/reps/coords_space.pickle','rb') as f:
    coords_space = pickle.load(f)
with open('../data/reps/prot_labels.pickle','rb') as f:
    prot_labels = pickle.load(f)

#SA.init_shape_space(coords_space)
#effective_dim_coords = effective_dim_SRV(coords_space, SA)
#frechet_radius_coords = frechet_radius(coords_space, SA)[1]
def run_model(placeholder=0):
    for n, mod in enumerate(models):
        SA = ShapeAnalysis(res = n_res[n], spline_order=sp_order[n], id = n)
        SA.init_shape_space(coords_space)
        effective_dim_coords = effective_dim_SRV(coords_space, SA)
        frechet_radius_coords = frechet_radius(coords_space, SA)[1]
        n_layers = mod()[0].num_layers
        with open('../data/reps/coords_esm_space_'+model_names[n]+'_.pickle','rb') as f:
            coords_esm_space = pickle.load(f)
        #coords_esm_space = [[] for i in range(n_layers)]
        #dmats[n] = [np.zeros([len(prot_labels),len(prot_labels)]) for _ in range(n_layers)]
        shape_spaces = []
        for layer in tqdm(range(n_layers)):
            #coords_esm_space[layer] = load_representations(model_names[n]+'_k',layer)
            new_shape_space = ShapeAnalysis(res=n_res[n],spline_order=sp_order[n],id=(n+1)*100+layer)
            new_shape_space.init_shape_space(coords_esm_space[layer])
            shape_spaces.append(new_shape_space)
            effective_dims[n].append(effective_dim_SRV(coords_esm_space[layer], shape_spaces[layer]))
            frechet_radii[n].append(frechet_radius(coords_esm_space[layer], shape_spaces[layer])[1])
            coords_esm_space[layer] = [] #delete data to save space

        fig, ax = plt.subplots(1,2,figsize=(6,3))
        #ax[0].plot(max_raidii[n][0],'o-',color='brown')
        #ax[0].axhline(y=max_radius_coords,color='k',linestyle='--')
        #ax[0].set_title('Max Radius')
        ax[0].plot(effective_dims[n],'o-',color='brown')
        ax[0].axhline(y=effective_dim_coords,color='k',linestyle='--')
        ax[0].grid('on')
        ax[0].set_title('Effective Dimension')
        ax[1].plot(frechet_radii[n],'o-',color='brown')
        ax[1].axhline(y=frechet_radius_coords,color='k',linestyle='--')
        ax[1].grid('on')
        ax[1].set_title('Frechet Radius')
        fig.tight_layout()
        #fig.savefig('../figures/shape_summary_'+model_names[n]+'_order_2_.png',dpi=300)
        np.save(f'../data/eff_dim_{n_res[n]}_{sp_order[n]}.npy',effective_dims[n])
        np.save(f'../data/frechet_rad_{n_res[n]}_{sp_order[n]}.npy', frechet_radii[n])
        print('Figure saved')

if __name__=="__main__":
    p = multiprocessing.Process(target=run_model,args=(0,))
    p.start()

