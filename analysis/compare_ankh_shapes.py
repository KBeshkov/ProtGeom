import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colormaps
import ankh
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


SA = ShapeAnalysis()
model_names = ['ankh']
models = [ankh]
dmats = [[] for _ in range(len(models))]
frechet_radii = []
effective_dims = []
colormap = colormaps['tab10']
prot_types = ['a','b','c','d','e','f','g','k']

with open('../data/reps/coords_space.pickle','rb') as f:
    coords_space = pickle.load(f)
with open('../data/reps/prot_labels.pickle','rb') as f:
    prot_labels = pickle.load(f)

SA.init_shape_space(coords_space)
effective_dim_coords = effective_dim_SRV(coords_space, SA)
frechet_radius_coords = frechet_radius(coords_space, SA)[1]
def run_model(placeholder=0):
    shape_spaces = []
    with open(f'../data/reps/coords_ankh_space_ankh_k.pickle','rb') as f:
        coords_ankh_space = pickle.load(f)
    n_layers = len(coords_ankh_space)-1
    for layer in tqdm(range(n_layers)):
        #layer = 2*l
        new_shape_space = ShapeAnalysis(id=layer)
        new_shape_space.init_shape_space(coords_ankh_space[layer])
        shape_spaces.append(new_shape_space)
        effective_dims.append(effective_dim_SRV(coords_ankh_space[layer], shape_spaces[layer]))
        frechet_radii.append(frechet_radius(coords_ankh_space[layer], shape_spaces[layer])[1])

    fig, ax = plt.subplots(1,2,figsize=(6,3))
    ax[0].plot(effective_dims,'o-',color='brown')
    ax[0].axhline(y=effective_dim_coords,color='k',linestyle='--')
    ax[0].grid('on')
    ax[0].set_title('Effective Dimension')
    ax[1].plot(frechet_radii,'o-',color='brown')
    ax[1].axhline(y=frechet_radius_coords,color='k',linestyle='--')
    ax[1].grid('on')
    ax[1].set_title('Frechet Radius')
    fig.tight_layout()
    fig.savefig('../figures/shape_summary_ankh.png',dpi=300)
    print('Figure saved')

if __name__=="__main__":
    p = multiprocessing.Process(target=run_model,args=(0,))
    p.start()
