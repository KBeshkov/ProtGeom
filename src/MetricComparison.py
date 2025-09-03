from persim import gromov_hausdorff, plot_diagrams
from ripser import ripser as tda
from sklearn.metrics import pairwise_distances
import numpy as np

class MetricSpaceComparison:
    def __init__(self, pclouds_1, pclouds_2, epsilon_filtration = np.linspace(0,1,10),normalize=False):
        self.pclouds_1 = pclouds_1
        self.pclouds_2 = pclouds_2
        self.epsilon_filtration = epsilon_filtration #Either a percentile specifying percentiles of distance distribution or integer specifying number of neighbors
        if normalize:
            self.normalize_point_clouds()

    def normalize_point_clouds(self):
        """Normalize point clouds to have zero mean and unit variance."""
        for i in range(len(self.pclouds_1)):
            if not isinstance(self.pclouds_1[i], np.ndarray):
                raise ValueError("Point clouds must be numpy arrays.")
            self.pclouds_1[i] = (self.pclouds_1[i] - np.mean(self.pclouds_1[i], axis=0)) / np.std(self.pclouds_1[i], axis=0)
        
        for i in range(len(self.pclouds_2)):
            if not isinstance(self.pclouds_2[i], np.ndarray):
                raise ValueError("Point clouds must be numpy arrays.")
            self.pclouds_2[i] = (self.pclouds_2[i] - np.mean(self.pclouds_2[i], axis=0)) / np.std(self.pclouds_2[i], axis=0)

    def euclidean_metric(self):
        """Compute the Euclidean distance between all point clouds in both metric spaces."""
        dmats_1 = []
        dmats_2 = []
        for pcloud in self.pclouds_1:
            if not isinstance(pcloud, np.ndarray):
                raise ValueError("Point clouds must be numpy arrays.")
            dmats_1.append(pairwise_distances(pcloud, metric='euclidean'))
            
        for pcloud in self.pclouds_2:
            if not isinstance(pcloud, np.ndarray):
                raise ValueError("Point clouds must be numpy arrays.")
            dmats_2.append(pairwise_distances(pcloud, metric='euclidean'))
        return dmats_1, dmats_2
    
    def epsilon_metric(self, dmats, epsilon):
        """Compute an epsilon (percentile) distance between all point clouds in both metric spaces. Also returns epsilon contact maps."""
        contact_maps = []
        for dmat in dmats:
            if epsilon % 1 == 0:
                k_neighbors = int(epsilon)
            else:
                k_neighbors = int((len(dmat)-1)*epsilon)
            contact_map = np.zeros(dmat.shape)
            sorted_indices = np.argsort(dmat, axis=1)
            for i in range(len(dmat)):
                neighbors = sorted_indices[i, 1:k_neighbors+1]
                contact_map[i, neighbors] = 1
            contact_maps.append(contact_map)
        return contact_maps

    def compute_gromov_hausdorff_filtration(self):
        """Compute the Gromov-Hausdorff distance between all pairs of proteins for each epsilon in the filtration."""
        GH_distances = [[] for _ in range(len(self.epsilon_filtration))]
        dmats_1, dmats_2 = self.euclidean_metric()
        for i, epsilon in enumerate(self.epsilon_filtration):
            contact_maps_1 = self.epsilon_metric(dmats_1, epsilon)
            contact_maps_2 = self.epsilon_metric(dmats_2, epsilon)
            for j in range(len(self.pclouds_1)):
                GH_distances[i].append(gromov_hausdorff(contact_maps_1[j], contact_maps_2[j]))
        return GH_distances
    
    def compute_Hamming_filtration(self):
        """Compute the graph filtration for each point cloud in both metric spaces."""
        Hamming_distances = [[] for _ in range(len(self.epsilon_filtration))]
        dmats_1, dmats_2 = self.euclidean_metric()
        for i, epsilon in enumerate(self.epsilon_filtration):
            contact_maps_1 = self.epsilon_metric(dmats_1, epsilon)
            contact_maps_2 = self.epsilon_metric(dmats_2, epsilon)
            for j in range(len(self.pclouds_1)):
                v = len(contact_maps_1[j]) #number of vertices
                e = int(v * epsilon) #number of edges
                #norm_correction = 2*e*v*(1-e/((v-1))) #correction factor by the first moment of the hypergeometric distribution
                Hamming_distances[i].append(np.sum(contact_maps_1[j] != contact_maps_2[j]))#/(norm_correction))
        return Hamming_distances