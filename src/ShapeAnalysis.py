import numpy as np
import scipy.linalg
from scipy.interpolate import make_interp_spline
import geomstats.backend as gs
from geomstats.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOrigin,
    SRVMetric,
)
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from tqdm import tqdm

class ShapeAnalysis:
    def __init__(self,res=1000,spline_order=2,id=0):
        self.res = res
        self.domain = np.linspace(0,1,self.res)
        self.spline_order = spline_order
        self.id = id
    
    def interpolate_curve(self, y):
        t = np.linspace(0,1,len(y))
        b_t = make_interp_spline(t, y,k=self.spline_order)
        return b_t
    
    def apply_transforms_curve(self,y):
        curve = self.interpolate_curve(y)(self.domain)
        curve = self.space.projection(curve)
        curve = self.space.normalize(curve)
        return curve
    
    def shape_analysis(self,X,Y):
        self.space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=max(len(X.T),len(X.T)), k_sampling_points=self.res, equip=False)
        self.space.equip_with_metric(SRVMetric)

        self.space.equip_with_group_action(("rotations"))
        self.space.equip_with_quotient()
        return self.compute_metric(X, Y)
    
    def init_shape_space(self, data):
        self.space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=max(len(data[0].T),len(data[0].T)), k_sampling_points=self.res, equip=False)
        self.space.equip_with_metric(SRVMetric)
        self.space.equip_with_group_action(("rotations"))
        self.space.equip_with_quotient()
        return self.space
    
    def compute_metric(self, X, Y):
        'Same as shape_analysis except it avoids redefining the shape space'
        curve_a = self.apply_transforms_curve(X)
        curve_b = self.apply_transforms_curve(Y)
        return self.space.quotient.metric.dist(curve_a, curve_b)


def frechet_radius(pclouds, shape_space):
    interpolated_pclouds = np.stack([shape_space.apply_transforms_curve(pcloud)
                                     for pcloud in pclouds])
    mean = FrechetMean(shape_space.space)
    mean.fit(interpolated_pclouds)
    frechet_mean = mean.estimate_
    distances = np.zeros(len(pclouds))
    for i,pcloud in enumerate(interpolated_pclouds):
        try:
            distances[i] = shape_space.compute_metric(frechet_mean, pcloud)
        except:
            continue
    return frechet_mean, np.nanmean(distances)

def effective_dim_SRV(pclouds, shape_space):
    interpolated_pclouds = np.stack([shape_space.apply_transforms_curve(pcloud)
                                     for pcloud in pclouds])
    tpca = TangentPCA(shape_space.space)
    tpca.fit(np.stack(interpolated_pclouds))
    eigvals = tpca.explained_variance_
    effective_dim = (np.sum(eigvals)**2)/np.sum(eigvals**2)
    return effective_dim

def effective_dim(pclouds,res=1000):
    SA = ShapeAnalysis(res=res)
    interpolated_pclouds = np.stack([SA.interpolate_curve(pcloud)(SA.domain) for pcloud in pclouds])
    interpolated_pclouds = interpolated_pclouds.reshape(len(pclouds),-1)
    pca_emb = PCA().fit(interpolated_pclouds)
    effective_dim = (np.sum(pca_emb.explained_variance_)**2)/np.sum(pca_emb.explained_variance_**2)
    return effective_dim

def compute_dmat(data, shape_space, subsamples=1,n_samples=100):
    dmats = []
    for i in range(subsamples):
        samples = np.random.choice(np.arange(0,len(data)),n_samples,replace=True)
        data_sub = data[samples]
        dmat = np.zeros([n_samples,n_samples])
        for i in tqdm(range(len(data_sub))):
            for j in range(len(data_sub)):
                if i<j:
                    try:
                        dmat[i,j] = shape_space.shape_analysis(data_sub[i],data_sub[j])
                    except:
                        dmat[i,j] = np.nan
        dmat[np.where(np.isnan(dmat))[0]] = np.nanmean(dmat)
        dmat = dmat + dmat.T
        dmats.append(dmat)
    return dmats
            
