## Installation

### Prerequisites
We have used a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) environment that can be initialized through the environment.yml file.

### Steps
1.  **Clone the repository:**
    ```
    git clone https://github.com/KBeshkov/ProtGeom.git
    cd ProtGeom
    ```

2.  **Create and activate the environment:**
    ```
    conda env create -f environment.yml
    conda activate protgeom
    ```

3.  **Install the package**
    ```
    pip install -e .
    ```


## Usage

In order to create folders and the pdb files run **sample_scop_proteins.py** from the /tools/ folder. 

Afterwards **get_representations.py** will generate files with the desired representations in the /data/ folder.

Once these two scripts run successfully, the scripts in the /analysis/ folder can be executed to generate figures like those in the paper. For each script the user can edit the list of models that they want to analyze.

## Citation


```bibtex
@inproceedings{beshkov2026towards,
  title={Towards Understanding the Shape of Representations in Protein Language Models},
  author={Beshkov, Kosio and Malthe-S{\o}renssen, Anders},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
> [https://openreview.net/pdf?id=Dnn8SSBJaY]

## License
This project is licensed under the terms of the **MIT** license.
