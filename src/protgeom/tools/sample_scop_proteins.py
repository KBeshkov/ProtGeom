import pandas as pd
import numpy as np
from pathlib import Path
current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent.parent

def scopcla_to_dict(file):
    column_names = ["SID", "PDB_ID", "Chain_ID", "SCOPe class", "SUNID", "misc"]

    df = pd.read_csv(file, delim_whitespace=True, header=None, names=column_names)
    return df

def sample_class(class_id, n_samples=2):
    data_file = root_dir / "data" / "dir.cla.scope.txt"
    scop_dict = scopcla_to_dict(data_file)
    class_entries = scop_dict['SCOPe class'].str.startswith(class_id).values
    pdb_entries = scop_dict['PDB_ID'].values[class_entries]
    if n_samples == 0:
        n_samples = len(pdb_entries)
    return pdb_entries[np.random.choice(np.arange(0,len(pdb_entries)),replace=False,size=n_samples)]
    

from Bio.PDB import PDBList

def download_pdbs(pdb_ids, out_dir=root_dir / "data" / "pdbs", file_format="mmCif",subfolder='a'):
    """
    Downloads PDB files from RCSB using Biopython.
    
    Parameters:
    - pdb_ids: list of 4-letter PDB codes (e.g. ['1abc', '2xyz'])
    - out_dir: directory to save downloaded PDB files
    - file_format: "pdb" or "mmCif"
    """
    pdbl = PDBList(server="https://files.rcsb.org")
    for pdb_id in pdb_ids:
        pdbl.retrieve_pdb_file(pdb_id, pdir=out_dir / subfolder, file_format=file_format, overwrite=True)
        
if __name__ == '__main__':
    prot_classes = ['a','b','c','d','e','f','g','k']
    for prot_class in prot_classes:
        try:
            prot_ids = list(sample_class(prot_class))
        except:
            prot_ids = sample_class(prot_class,n_samples=0) #samples all available proteins
        download_prots = download_pdbs(prot_ids,subfolder=prot_class)
