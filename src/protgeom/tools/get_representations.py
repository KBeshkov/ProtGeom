import argparse
import multiprocessing
import pickle
from os import walk
from pathlib import Path

import ankh
import esm
import numpy as np
import torch
from Bio import PDB
from tqdm import tqdm

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent.parent
prot_dir = root_dir / "data" / "pdbs"

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

# family, loader
MODEL_REGISTRY = {
    'esm2_t6_8M_UR50D':    ('esm',  esm.pretrained.esm2_t6_8M_UR50D),
    'esm2_t12_35M_UR50D':  ('esm',  esm.pretrained.esm2_t12_35M_UR50D),
    'esm2_t30_150M_UR50D': ('esm',  esm.pretrained.esm2_t30_150M_UR50D),
    'esm2_t33_650M_UR50D': ('esm',  esm.pretrained.esm2_t33_650M_UR50D),
    'ankh_base':           ('ankh', ankh.load_base_model),
    'ankh_large':          ('ankh', ankh.load_large_model),
}


def parse_chain(path):
    """Return (CA coords, single-letter sequence) for chain A. None if absent."""
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", path)
    if 'A' not in structure[0]:
        return None
    coords, protein = [], ''
    for residue in structure[0]['A']:
        if 'CA' not in residue or residue.resname not in d3to1:
            continue
        coords.append(residue['CA'].coord)
        protein += d3to1[residue.resname]
    if not coords:
        return None
    return np.array(coords), protein


def embed_esm(model, alphabet, protein, layers):
    # ESM2 tokens: [BOS, aa_0, ..., aa_{L-1}, EOS] -> residues are at [1 : 1+L].
    _, _, tokens = alphabet.get_batch_converter()([('p', protein)])
    with torch.no_grad():
        out = model(tokens, repr_layers=layers)
    return [out["representations"][l][0, 1 : 1 + len(protein)].cpu().numpy()
            for l in layers]


def embed_ankh(model, tokenizer, protein, layers):
    # Ankh (T5-encoder) tokens: [aa_0, ..., aa_{L-1}, EOS] -> residues are at [:L].
    enc = tokenizer.batch_encode_plus(
        [list(protein)],
        add_special_tokens=True,
        padding=True,
        is_split_into_words=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
    return [out.hidden_states[l][0, : len(protein)].cpu().numpy() for l in layers]


def run_model(model_name):
    family, loader = MODEL_REGISTRY[model_name]
    model, tok = loader()
    model.eval()
    n_layers = model.num_layers if family == 'esm' else model.config.num_layers
    layers = list(range(n_layers + 1))

    subfolders = sorted(next(walk(prot_dir))[1])
    last_sub = subfolders[-1]

    coords_space = []
    prot_labels = []
    rep_space = [[] for _ in layers]

    for count, sub in enumerate(subfolders):
        print(sub)
        filenames = next(walk(prot_dir / sub), (None, None, []))[2]
        for fname in tqdm(filenames):
            if not fname.endswith('cif'):
                continue
            parsed = parse_chain(prot_dir / sub / fname)
            if parsed is None:
                continue
            coords, protein = parsed
            if family == 'esm':
                per_layer = embed_esm(model, tok, protein, layers)
            else:
                per_layer = embed_ankh(model, tok, protein, layers)
            coords_space.append(coords)
            prot_labels.append(count)
            for l, r in enumerate(per_layer):
                rep_space[l].append(r)

    out_dir = root_dir / "data" / "reps"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "coords_ankh_space" if family == "ankh" else "coords_esm_space"
    with open(out_dir / "coords_space.pickle", 'wb') as f:
        pickle.dump(coords_space, f)
    with open(out_dir / f"{prefix}_{model_name}_{last_sub}.pickle", 'wb') as f:
        pickle.dump(rep_space, f)
    with open(out_dir / "prot_labels.pickle", 'wb') as f:
        pickle.dump(prot_labels, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["esm2_t6_8M_UR50D"],
                    choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--parallel", action="store_true",
                    help="run models in parallel processes (more RAM).")
    args = ap.parse_args()

    if args.parallel:
        procs = [multiprocessing.Process(target=run_model, args=(m,))
                 for m in args.models]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    else:
        for m in args.models:
            run_model(m)
