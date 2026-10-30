# Role-aware line-graph GNNs for microbial interaction prediction

Code for the paper *Interpretable Graph Neural Networks for Microbial Interaction Prediction*.

Code for predicting pairwise microbial interactions with graph neural networks (GNNs) on a **role-aware line graph**.
Each node is one directional measurement (effector → affected species, under one carbon condition). Two nodes are connected
if, under the same carbon condition, they share the same effector or the same affected species.

Authors: Elham Gholamzadeh, Kajal Singla, Nico Scherf (Max Planck Institute for Human Cognitive and Brain Sciences / ScaDS.AI, Leipzig).

## Notebooks

| Notebook | Task | Models |
|---|---|---|
| `oneway_roleaware.ipynb` | One-way sign prediction (positive / negative effect of one species on another) | GAT-SAGE, GAT, GraphSAGE, CGNN, and GAT-SAGE on the role-agnostic graph (ablation) |
| `twoway_roleaware.ipynb` | Two-way interaction type (mutualism / competition / parasitism) | Same models as above |
| `imbalance_roleaware.ipynb` | Class imbalance in one-way (or two-way) prediction | GAT-SAGE (or GraphSAGE) with mixed entropy minimisation for several balancing factors λ, and with random oversampling of training nodes |

Each notebook prints the LaTeX tables, paired t-tests across folds (GAT-SAGE vs. the other GNNs, role-aware vs. role-agnostic), and saves confusion matrices and t-SNE plots of the hidden-layer embeddings.

The XGBoost baseline reported in the paper follows Nestor et al. (2023), using their tuned hyperparameters on the same 13 features; see their repository: <https://github.com/einatnestor/Microbial-interaction-prediction>.

### Labels
- **One-way:** each directional measurement is labeled positive or negative from its measured effect; the label definition is set in the first cell of `oneway_roleaware.ipynb`.
- **Two-way:** each directional effect is positive if it is > 0. Mutualism = (+,+), competition = (−,−), parasitism = one positive and one negative effect.
- **Imbalance:** set `TASK` (one-way or two-way) and `MODEL` (GAT-SAGE or GraphSAGE) in the first cell of `imbalance_roleaware.ipynb`.

### Evaluation
- Grouped, stratified 5-fold cross-validation (`StratifiedGroupKFold`, fixed seed). Both directions of a co-culture (same species pair and carbon condition) are always in the same fold.
- All models use the same folds.
- Transductive setting: the full graph and all node features are available during training; metrics are computed on validation nodes only.
- Two-layer models, hidden size 100, Adam (learning rate 1e-3), 3,000 epochs, no dropout.

## Data

We use the pairwise interaction data of Nestor et al. (2023), available at
<https://github.com/einatnestor/Microbial-interaction-prediction>.

Place the data folder next to the notebooks:

```
role-aware-line-graph-microbial-interactions/
├── Data_msystems/
│   ├── Features.csv            # one-way data
│   └── Features_two_way.csv    # two-way data
├── oneway_roleaware.ipynb
├── twoway_roleaware.ipynb
└── imbalance_roleaware.ipynb
```

The water control is removed in every notebook, leaving 39 carbon conditions.

## Installation

```bash
pip install -r requirements.txt
```

- A GPU is strongly recommended. On CPU, 3,000 epochs × 5 folds × several models take many hours.
- For the GPU, install the CUDA builds of PyTorch and DGL that match your CUDA version (see <https://www.dgl.ai/pages/start.html>). The notebooks fall back to the CPU and print a warning if DGL cannot use the GPU.
- `GraphRicciCurvature` is only needed for CGNN. The curvature is computed once and cached in `results/`.

## Running

Open a notebook and run all cells (**Kernel → Restart & Run All**). Results are written to `results/` (one-way and two-way) and `results_imbalance/` (imbalance).

For a quick test, set the number of epochs with an environment variable before starting Jupyter:

```bash
EPOCHS=2 jupyter notebook
```

## Citation

If you use this code, please cite:

> Gholamzadeh E, Singla K, Scherf N. Predicting microbial interactions using graph neural networks. arXiv:2511.02038, 2025.

and the original data:

> Nestor E, Toledano G, Friedman J. Interactions between culturable bacteria are predicted by individual species' growth. mSystems 8(2):e00836-22, 2023.
