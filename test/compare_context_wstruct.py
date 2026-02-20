import torch
import matplotlib.pyplot as plt
from matplotlib import colormaps
import esm
import numpy as np
import sys
sys.path.append("../src/")
from dynamic_rep_loading import load_representations
import multiprocessing
from Bio import PDB
from tqdm import tqdm
from MetricComparison import MetricSpaceComparison
from os import walk
import pickle
from scipy import stats


prot_dir = '../data/pdbs/'
subfolders =  next(walk(prot_dir))[1]
subfolders.sort()

model_names = ['esm2_t6_8M_UR50D','esm2_t12_35M_UR50D',
               'esm2_t30_150M_UR50D']#,'esm2_t33_650M_UR50D']
models = [esm.pretrained.esm2_t6_8M_UR50D, esm.pretrained.esm2_t12_35M_UR50D, esm.pretrained.esm2_t30_150M_UR50D]# [esm.pretrained.esm2_t6_8M_UR50D,esm.pretrained.esm2_t12_35M_UR50D,
         # esm.pretrained.esm2_t30_150M_UR50D]#,esm.pretrained.esm2_t33_650M_UR50D]
with open('../data/reps/coords_space.pickle','rb') as f:
    coords_space = pickle.load(f)
with open('../data/reps/prot_labels.pickle','rb') as f:
    prot_labels = pickle.load(f)
epsilons = np.arange(1,20)
prot_types = np.unique(prot_labels)
prot_type_names = ['Alpha', 'Beta', 'Alpha/Beta', 'Alpha+Beta', 'Alpha and Beta', 'Peptides and Surface', 'Small Proteins', 'Designed']
colormap = colormaps['tab10']

print('Starting predictions')
def run_model(placeholder=0):
    for n, mod in enumerate(models):
        # n_layers = mod()[0].num_layers
        with open('../data/reps/coords_esm_space_'+model_names[n]+'_.pickle','rb') as f: #Change according to which model files you want to use
            coords_esm_space = pickle.load(f)
        n_layers = len(coords_esm_space)
        prot_lens = [prot.shape[1] for prot in coords_esm_space[0]]
        means_tensor = np.zeros([len(prot_type_names),n_layers,len(epsilons)])
        for layer in tqdm(range(n_layers)):
            #coords_esm_space[layer] = load_representations(model_names[n]+'_k', layer)
            hamming_distances = [[] for i in range(len(prot_types))]
            mean_comparison = []
            std_comparison = []
            ham_array = np.zeros([len(epsilons),len(prot_lens)])
            rand_array = np.zeros([len(epsilons),len(prot_lens)])
            idx_count = 0
            print(coords_esm_space[layer][1].shape)
            for idx in prot_types:
                new_prot_class = []
                new_esm_class = []
                for i in range(len(prot_labels)):
                    if (prot_labels==idx)[i]:
                        new_prot_class.append(coords_space[i])
                        new_esm_class.append(coords_esm_space[layer][i].squeeze())
                print(new_esm_class[0].shape)
                print(new_prot_class[0].shape)
                rand_pcloud = [np.random.rand(len(pcloud),coords_esm_space[layer][0].shape[1]) for pcloud in new_prot_class]
                Metric_class_rand = MetricSpaceComparison(new_prot_class, rand_pcloud, epsilons)
                Metric_class = MetricSpaceComparison(new_prot_class, new_esm_class, epsilons)
                hamming_distances[idx].append(Metric_class.compute_Hamming_filtration())
                rand_hamming_distances = Metric_class_rand.compute_Hamming_filtration()
        
                ham_array[:,idx_count:idx_count+len(new_prot_class)] = np.array(hamming_distances[idx])
                rand_array[:,idx_count:idx_count+len(new_prot_class)] = np.array(rand_hamming_distances)
                idx_count += len(new_prot_class)
        
                means = np.array([np.nanmean(hamarray) for hamarray in hamming_distances[idx][0]])
                stds = np.array([np.nanstd(hamarray)/2 for hamarray in hamming_distances[idx][0]])
        
                means_rand = np.array([np.nanmean(hamarray) for hamarray in rand_hamming_distances])
                stds_rand = np.array([np.nanstd(hamarray)/2 for hamarray in rand_hamming_distances])
        
                mean_comparison.append(means/means_rand)
                std_comparison.append(stds/stds_rand)
                means_tensor[idx,layer,:] = means/means_rand
        
            struct_sim = np.nanmin(ham_array/rand_array,0)
            # slope, intercept, r_value, p_value, std_err = stats.linregress(prot_lens,struct_sim)
            r_value = stats.spearmanr(prot_lens,struct_sim)[0]
            print('Done with model '+model_names[n]+' layer '+str(layer+1))
        
        
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(prot_lens, struct_sim,'.')
            # ax.plot(prot_lens,slope.item()*np.array(prot_lens)+intercept.item(),'k')
            ax.set_title(f'r={round(r_value,2)}')
            ax.grid('on')
            fig.savefig('../figures/structure_similarity_'+model_names[n]+'_layer_'+str(layer)+'.png',dpi=300,bbox_inches='tight')
        
            np.save('../data/structure_sim_'+model_names[n]+'_layer_'+str(layer)+'.npy', struct_sim)


if __name__=="__main__":
    p = multiprocessing.Process(target=run_model,args=(0,))
    p.start()


