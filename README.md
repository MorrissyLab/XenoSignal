# XenoSignal

Code and notebooks for the paper:
**“XenoSignal: Investigating Intra- and Inter-Species Ligand–Receptor Interactions Using AlphaFold3.”**
*Aly O. Abdelkareem, Courtney F. Hall, Athish Marutharaj, Kiran Narta, Theodore B. Verhey & A. Sorana Morrissy*
bioRxiv: [https://www.biorxiv.org/content/10.1101/2025.08.13.670200v1](https://www.biorxiv.org/content/10.1101/2025.08.13.670200v1)

---

## What this repo does

XenoSignal builds cross-species ligand–receptor (LR) pairs from CellChat interactions, retrieves the corresponding protein sequences/IDs, and prepares AlphaFold3 (AF3) jobs to model LR complexes. It then computes binding features (BindCraft, PRODIGY, ipSAE) and analyzes them to produce the manuscript’s supplementary figures and tables.

---

## Repository layout

```
XENOSIGNAL/
├─ Data/
│  ├─ cellchat_data/        # CellChat LR interactions (input)
│  ├─ orthologs/            # gene orthology tables (input)
│  └─ xenosignal_jobs.zip   # AF3 job package (output of N1)
├─ IPSAE/
│  ├─ ipsae.py
│  ├─ run_ipsae.ps1         # PowerShell runner for ipSAE
│  └─ README.md
├─ notebooks/
│  ├─ N1_CellChat_XenoSignal.ipynb
│  ├─ N2_binding_features_generate.ipynb
│  └─ N3_Binding_Features_Analysis.ipynb
├─ xeno/                    # Python helpers (I/O, features, plotting, utils)
│  ├─ biopython_utils.py
│  ├─ features.py
│  ├─ io.py
│  ├─ pl.py
│  └─ pyrosetta_utils.py    # optional utilities (not required for main workflow)
├─ LICENSE
└─ README.md
```

---

## Requirements

* **Python ≥ 3.10**
* Python packages: `pandas`, `polars`, `numpy`, `scipy`, `scikit-learn`, `biopython`, `matplotlib`
* Optional (for some utilities): `pyrosetta` (not required for reproducing the figures)
* **AlphaFold3** (external; supply your own AF3 binaries/resources)
* **BindCraft** and **PRODIGY** (called via wrappers under `./Xeno source code/`)
* **ipSAE** (invoked via `IPSAE/run_ipsae.ps1`; requires PowerShell on Windows or adapt for your OS)

> External tools (AF3, BindCraft, PRODIGY, ipSAE) are not bundled here. Please install/obtain them per their licenses and set paths accordingly.

---

## Quickstart

1. **Create environment**

   ```bash
   conda create -n xenosignal python=3.10 -y
   conda activate xenosignal
   pip install pandas polars numpy scipy scikit-learn biopython matplotlib
   ```

2. **Prepare inputs**

   * Place CellChat interaction tables under `./Data/cellchat_data/`.
   * Place ortholog mapping tables under `./Data/orthologs/`.
   * Ensure wrappers and binaries for BindCraft/PRODIGY are accessible under `./Xeno source code/`.
   * Make ipSAE available (e.g., run via `IPSAE/run_ipsae.ps1`).

3. **Run the notebooks in order**

   * **N1\_CellChat\_XenoSignal.ipynb**
     Builds cross-species LR pairs (mouse→human, human→mouse) using orthologs, resolves protein IDs/sequences, and **packages AF3 jobs** to `./Data/xenosignal_jobs.zip`.
   * **N2\_binding\_features\_generate.ipynb**
     Runs BindCraft & PRODIGY via `./Xeno source code/`, executes ipSAE via `IPSAE/run_ipsae.ps1`, parses outputs, and **merges features** to `./Data/binding_features.csv`.
   * **N3\_Binding\_Features\_Analysis.ipynb**
     Produces the manuscript visuals:

     * **Supplementary Figure 1:** clustering dendrogram + feature–feature correlation heatmap
     * **Supplementary Figure 2:** feature distributions colored by cluster assignment
       Saved to `./Figures/binding_features/`.

---

## Configuration tips

* If your tool paths aren’t on `PATH`, set an environment variable, e.g.:

  ```bash
  export XENO_TOOLS_DIR="/path/to/Xeno source code"
  ```
* AlphaFold3 inputs are generated as a zipped job bundle: `./Data/xenosignal_jobs.zip`. Unzip/submit to your AF3 infrastructure as appropriate.
* The ipSAE step can be run outside the notebook:

  ```powershell
  PowerShell -ExecutionPolicy Bypass -File IPSAE/run_ipsae.ps1
  ```

  Make sure it writes outputs where `N2_binding_features_generate.ipynb` expects them.

---



## Citing XenoSignal

If you use this code or figures, please cite:

> Abdelkareem AO, Hall CF, Marutharaj A, Narta K, Verhey TB, Morrissy AS.
> **XenoSignal: Investigating Intra- and Inter-Species Ligand–Receptor Interactions Using AlphaFold3.**
> bioRxiv (2025). doi:10.1101/2025.08.13.670200

**BibTeX**

```bibtex
@article{Abdelkareem2025XenoSignal,
  title   = {XenoSignal: Investigating Intra- and Inter-Species Ligand–Receptor Interactions Using AlphaFold3},
  author  = {Abdelkareem, Aly O. and Hall, Courtney F. and Marutharaj, Athish and Narta, Kiran and Verhey, Theodore B. and Morrissy, A. Sorana},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.08.13.670200},
  url     = {https://www.biorxiv.org/content/10.1101/2025.08.13.670200v1}
}
```

---

## License

See `LICENSE` in this repository.

---

## Contact

For questions or issues, please open a GitHub issue or contact the corresponding author listed in the manuscript.
