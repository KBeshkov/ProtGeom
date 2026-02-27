import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colormaps
import esm
import numpy as np
from protgeom.tools.dynamic_rep_loading import load_representations
from Bio import PDB
from protgeom.ShapeAnalysis import *
import multiprocessing
from os import walk
import pickle
from tqdm import tqdm
prot_dir = '../data/pdbs/'
subfolders =  next(walk(prot_dir))[1]
subfolders.sort()

n_res = [1000]
sp_order = [2]
SA = ShapeAnalysis(res=n_res[0],spline_order=sp_order[0],id=0)
model_names = ['esm2_t12_35M_UR50D','esm2_t12_35M_UR50D','esm2_t12_35M_UR50D','esm2_t12_35M_UR50D']#,'esm2_t12_35M_UR50D',
               #'esm2_t30_150M_UR50D']#,'esm2_t33_650M_UR50D']
models = [esm.pretrained.esm2_t12_35M_UR50D,esm.pretrained.esm2_t12_35M_UR50D,esm.pretrained.esm2_t12_35M_UR50D,esm.pretrained.esm2_t12_35M_UR50D]#, esm.pretrained.esm2_t12_35M_UR50D,
          #esm.pretrained.esm2_t30_150M_UR50D]#,esm.pretrained.esm2_t33_650M_UR50D]
colormap = colormaps['tab10']


with open('../data/reps/coords_space.pickle','rb') as f:
    coords_space = pickle.load(f)
with open('../data/reps/prot_labels.pickle','rb') as f:
    prot_labels = pickle.load(f)
prot_types = np.unique(prot_labels)
SA.init_shape_space(coords_space)
effective_dim_coords = effective_dim_SRV(coords_space, SA)
frechet_radius_coords = frechet_radius(coords_space, SA)[1]
def run_model(placeholder=0):
    for n, mod in enumerate(models):
        frechet_radii = np.zeros([len(prot_types), mod()[0].num_layers])
        effective_dims = np.zeros([len(prot_types), mod()[0].num_layers])
        n_layers = mod()[0].num_layers
        with open('../data/reps/coords_esm_space_'+model_names[n]+'_.pickle','rb') as f:
            coords_esm_space = pickle.load(f)
        #coords_esm_space = [[] for i in range(n_layers)]
        shape_spaces = []
        for layer in tqdm(range(n_layers)):
            for ix,idx in enumerate(prot_types):
                new_prot_class = []
                new_esm_class = []
                for i in range(len(prot_labels)):
                    if (prot_labels==idx)[i]:
                        new_prot_class.append(coords_space[i])
                        new_esm_class.append(coords_esm_space[layer][i])
                #coords_esm_space[layer] = load_representations(model_names[n]+'_k',layer)
                new_shape_space = ShapeAnalysis(res=n_res[n],spline_order=sp_order[n],id=n*100+layer*10 + ix)
                new_shape_space.init_shape_space(new_esm_class)
                shape_spaces.append(new_shape_space)
                effective_dims[ix,layer] = effective_dim_SRV(new_esm_class, shape_spaces[-1])
                frechet_radii[ix,layer] = frechet_radius(new_esm_class, shape_spaces[-1])[1]
            coords_esm_space[layer] = [] #delete data to save space

        #fig.savefig('../figures/shape_summary_'+model_names[n]+'_order_2_.png',dpi=300)
        np.save(f'../data/eff_dim_{idx}.npy',effective_dims)
        np.save(f'../data/frechet_rad_{idx}.npy', frechet_radii)

if __name__=="__main__":
    p = multiprocessing.Process(target=run_model,args=(0,))
    p.start()

