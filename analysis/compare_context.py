import torch
import matplotlib.pyplot as plt
from matplotlib import colormaps
import esm
import numpy as np
from protgeom.tools.dynamic_rep_loading import load_representations
import multiprocessing
from Bio import PDB
from tqdm import tqdm
from protgeom.MetricComparison import MetricSpaceComparison
from os import walk
import pickle
prot_dir = '../data/pdbs/'
subfolders =  next(walk(prot_dir))[1]
subfolders.sort()

model_names = ['esm2_t33_650M_UR50D']#['esm2_t6_8M_UR50D','esm2_t12_35M_UR50D',
              # 'esm2_t30_150M_UR50D']#,'esm2_t33_650M_UR50D']
models = [esm.pretrained.esm2_t33_650M_UR50D]# [esm.pretrained.esm2_t6_8M_UR50D,esm.pretrained.esm2_t12_35M_UR50D,
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
        n_layers = mod()[0].num_layers
        with open('../data/reps/coords_esm_space_esm2_t33_650M_UR50D_k.pickle','rb') as f:
            coords_esm_space = pickle.load(f)
        means_tensor = np.zeros([len(prot_type_names),n_layers,len(epsilons)])
        for layer in tqdm(range(n_layers)):
            #coords_esm_space[layer] = load_representations(model_names[n]+'_k', layer)
            hamming_distances = [[] for i in range(len(prot_types))]
            mean_comparison = []
            std_comparison = []
            print(coords_esm_space[layer][1].shape)
            for idx in prot_types:
                new_prot_class = []
                new_esm_class = []
              for i in range(len(prot_labels)):
                    if (prot_labels==idx)[i]:
                        new_prot_class.append(coords_space[i])
                        new_esm_class.append(coords_esm_space[layer][i])
                rand_pcloud = [np.random.rand(len(pcloud),coords_esm_space[layer][0].shape[1]) for pcloud in new_prot_class]
                Metric_class_rand = MetricSpaceComparison(new_prot_class, rand_pcloud, epsilons)
                Metric_class = MetricSpaceComparison(new_prot_class, new_esm_class, epsilons)
                hamming_distances[idx].append(Metric_class.compute_Hamming_filtration())
                rand_hamming_distances = Metric_class_rand.compute_Hamming_filtration()

                best_prot = np.argmin(np.min(np.array(hamming_distances[idx][0]),0))
                print('Best protein index in class: = ' + str(best_prot)+' : ' + str(np.min(np.array(hamming_distances[idx][0]),0)[best_prot]))
            
                means = np.array([np.nanmean(hamarray) for hamarray in hamming_distances[idx][0]])
                stds = np.array([np.nanstd(hamarray)/2 for hamarray in hamming_distances[idx][0]])
            
                means_rand = np.array([np.nanmean(hamarray) for hamarray in rand_hamming_distances])
                stds_rand = np.array([np.nanstd(hamarray)/2 for hamarray in rand_hamming_distances])
        
                mean_comparison.append(means/means_rand)
                std_comparison.append(stds/stds_rand)
                means_tensor[idx,layer,:] = means/means_rand

            print('Done with model '+model_names[n]+' layer '+str(layer+1))
            
            fig, ax = plt.subplots(figsize=(6, 6))
            for i, mean_comp in enumerate(mean_comparison):
                ax.plot(epsilons, mean_comp,'-o',markersize=2.5)
                #ax.fill_between(epsilons, mean_comp - std_comparison[i]/20, mean_comp + std_comparison[i]/20, alpha=0.2)
            ax.legend(prot_type_names,loc=4,prop={'size': 8})
            #plt.vlines([2,8],[0.6,0.6],[1,1],'k',alpha=0.5)
            ax.grid('on')
            ax.set_xlabel('Epsilon')
            ax.set_ylabel('Normalized Hamming Distance')
            ax.set_title(model_names[n]+' Layer '+str(layer+1))
            fig.savefig('../figures/mean_comparison_'+model_names[n]+'_layer_'+str(layer)+'.png', dpi=300, bbox_inches='tight')

        fig, axs = plt.subplots(1, len(prot_type_names)+1, figsize=(15, 3))
        for i in range(len(prot_type_names)):
            alphas = np.linspace(0.1,1,n_layers)
            for l in range(n_layers):
                axs[i].plot(epsilons, means_tensor[i,l,:],lw=1,color=colormap(i),alpha=alphas[l])
            axs[i].set_title(prot_type_names[i], fontsize=6)
            axs[i].set_xlabel('neighbors')
            axs[i].grid('on')

        for i in range(len(prot_type_names)):
            axs[len(prot_type_names)].plot(np.arange(1,n_layers+1),np.min(means_tensor[i],axis=1))
        axs[len(prot_type_names)].grid('on')
        axs[len(prot_type_names)].set_xlabel('layers')
        fig.tight_layout()

        fig.savefig('../figures/mean_comparison_'+model_names[n]+'.png', dpi=300, bbox_inches='tight')

if __name__=="__main__":
    p = multiprocessing.Process(target=run_model,args=(0,))
    p.start()

