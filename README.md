# AANRI Project

Comprehensive single-cell RNA-sequencing (scRNA-seq) analysis pipeline for cell type annotation, quality control, and cellular characterization across multiple brain regions.

## Overview

The AANRI project performs end-to-end analysis of scRNA-seq data from brain samples including:
- **DLPFC** (Dorsolateral Prefrontal Cortex)
- **Hippocampus** (hippo)
- **Caudate**
- **LIBD** (Lieber Institute for Brain Development) dataset integration

The pipeline includes sample processing, quality filtering, doublet detection, batch integration, cell type annotation, and comprehensive visualization and analysis workflows.

## Features

### 🧬 Core Functionality

- **Data Loading & Preprocessing**
  - Read 10x Genomics H5 and H5AD formats
  - Cell/gene filtering with customizable thresholds
  - Mitochondrial and ribosomal gene tagging
  - Quality metrics calculation (n_genes, n_counts, pct_counts_mt, pct_counts_ribo)

- **Quality Control**
  - Doublet detection using Scrublet
  - Batch effect removal using Harmony
  - Outlier detection and filtration
  - Batch purity assessment

- **Cell Type Annotation**
  - scPoli model-based annotation with reference datasets
  - Label transfer consensus calling
  - Uncertainty quantification
  - Multi-resolution annotation support

- **Clustering & Dimensionality Reduction**
  - scVI (single-cell Variational Inference) integration
  - Leiden clustering at multiple resolutions (0.5, 1.0, 1.5, 2.0, 2.5)
  - UMAP visualization with centroid labels
  - Harmony-based batch correction for integration

- **Analysis Utilities**
  - Barcode overlap analysis with statistical testing
  - Cell type entropy calculation
  - Donor/sample entropy metrics
  - Marker gene-based subset selection
  - Cell type purity assessment

### 📊 Visualization Tools

- **Batch comparison plots**: Side-by-side UMAP visualization
- **Category panels**: Multi-panel grid layouts showing cell type distributions
- **UMAP with centroids**: Automated label positioning with overlap prevention
- **Entropy heatmaps**: Visualization of clustering quality and contamination
- **PowerPoint export**: Automated figure export to PPTX format
- **Interactive batch analysis**: Per-sample quality assessment

### 🔧 Advanced Features

- **SEACells integration**: Metacell creation for aggregated analysis
- **Harmony batch correction**: Multi-resolution batch integration
- **scVI models**: Trained on reference datasets for transfer learning
- **Barcode purity checking**: Duplicate cell detection between samples
- **Google Drive integration**: Automated result upload

## Installation

### Requirements

```bash
Python 3.8+
```

### Dependencies

```python
anndata
scanpy
scvi-tools
scarches  # for scPoli models
SEACells
scrublet  # doublet detection
pandas
numpy
scipy
matplotlib
seaborn
adjustText
pydrive2  # Google Drive integration
python-pptx  # PowerPoint generation
```

### Setup

```bash
# Clone the repository
git clone https://github.com/dkim2197/AANRI_project.git
cd AANRI_project

# Install dependencies (recommended to use conda/mamba)
pip install -r requirements.txt
```

## Usage

### Basic Analysis Workflow

```python
import scanpy as sc
from AANRI_celltype_annotation_all import *

# Load data
adata = sc.read_10x_h5("path/to/cellbender_output_filtered.h5")

# Run standard preprocessing
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=400)

# Calculate QC metrics
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], inplace=True)

# Filter outliers
adata = adata[
    (~adata.obs["mt_outlier"]) & 
    (~adata.obs["ribo_outlier"])
].copy()

# Doublet detection
sc.pp.scrublet(adata)
adata = adata[~adata.obs.predicted_doublet].copy()
```

### Cell Type Annotation

```python
# Load trained scPoli model
from scarches.models.scpoli import scPoli

scpoli_model = scPoli.load("path/to/reference_model")
annotation, uncertainty = run_scpoli(
    scpoli_model, 
    reference_adata, 
    query_adata, 
    celltype='Celltype_reference'
)

# Add annotations to AnnData object
adata.obs['Celltype_annotated'] = annotation
adata.obs['Celltype_uncertainty'] = uncertainty
```

### Integration & Clustering

```python
# Select highly variable genes
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)

# scVI integration
import scvi
scvi.model.SCVI.setup_anndata(adata, batch_key='Batch')
model = scvi.model.SCVI(adata)
model.train(max_epochs=400, early_stopping=True)

# Get latent representation
adata.obsm["X_scVI"] = model.get_latent_representation()

# Clustering
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.leiden(adata, resolution=0.5)
sc.tl.umap(adata)
```

### Visualization

```python
# Plot UMAP with cell type annotations
plot_umap_with_centroids(
    adata, 
    color='Celltype_annotated', 
    annotate=True
)

# Panel plots for each category
plot_category_panels(
    adata, 
    category_column='AgeGroup',
    max_cols=5
)

# Side-by-side comparison
fig = plot_umap_side_by_side(
    adatas=[adata1, adata2],
    colors=['Celltype', 'Celltype'],
    titles=['Sample 1', 'Sample 2']
)
```

## Data Structure

### Input Data
- **Format**: 10x Genomics (`.h5`), AnnData (`.h5ad`)
- **Location**: `/mnt/pv_compute/dongsan/datasets/AANRI/NOVASEQ_FY*/`
- **Subdirectories**: Organized by fiscal year (FY23, FY24, FY25) and region (DLPFC, hippo, caudate)

### Output Structure
```
h5ad_per_sample_filtered_FINAL_FY23_24_25/
├── {Region}.txt                    # List of processed files
├── {Region}_harmony.h5ad           # Integrated dataset
├── {Region}_scvi/                  # scVI model and data
├── {Region}_scvi_top200/           # scVI with marker genes
└── integrated_filtered_final/
    └── {Region}_adata_consensus_full.h5ad  # Final consensus annotation
```

## Key Functions

### `run_SEACells(adata)`
Performs SEACells metacell aggregation with scVI latent representation.

### `compute_barcode_overlap_with_pval_and_filter(adata_list, ...)`
Detects duplicate cells across samples using barcode overlap with hypergeometric test.

### `cluster_entropy(adata, cluster_key, label_key, ...)`
Calculates Shannon entropy for cluster purity assessment.

### `run_scpoli(scpoli_model, source_adata, target_adata, celltype)`
Performs label transfer using scPoli model with uncertainty quantification.

### `plot_umap_with_centroids(adata, color, annotate=True)`
Creates UMAP plots with automatic centroid labeling and overlap prevention.

### `export_figure_to_ppt(fig, ppt_path, slide_title, ...)`
Exports matplotlib figures to PowerPoint presentations.

## Configuration

### Region-Specific Settings

Edit the `Region` variable to analyze different brain areas:

```python
Region = 'DLPFC'   # Dorsolateral Prefrontal Cortex
Region = 'hippo'   # Hippocampus
Region = 'caudate' # Caudate
Region = 'LIBD'    # LIBD dataset
```

### Filtering Thresholds

Customize quality control parameters:

```python
# Mitochondrial/ribosomal filtering
mt_threshold = 5      # percent_counts_mt
ribo_threshold = 10   # percent_counts_ribo

# Gene/cell filtering
min_genes = 400
min_cells = 3

# Doublet detection
scrublet_threshold = 0.5
```

### Entropy-Based Filtering

```python
donor_thresh = 0.8      # Donor entropy threshold
ctype_thresh = 2        # Cell type entropy threshold
```

## Sample Metadata

Metadata is loaded from:
- `/mnt/pv_compute/dongsan/datasets/AANRI/sample_inf.xlsx` (AANRI samples)
- `/mnt/pv_compute/dongsan/datasets/AANRI/LIBD_DLPFC/LIBD_DLPFC_CAUC_info.xlsx` (LIBD samples)
- `/mnt/pv_compute/dongsan/datasets/AANRI/global_ancestry.xlsx` (Ancestry information)

Metadata includes:
- `BrNum`: Brain specimen number
- `AgeDeath`: Age at death
- `Sex`: Biological sex
- `PMI`: Post-mortem interval
- `AgeGroup`: Age bracket categorization
- `YRI`, `CEU`, `CHB`: Global ancestry proportions

## Reference Models

Trained scPoli models are stored at:
```
/mnt/pv_compute/dongsan/datasets/AANRI/model_scpoli/
├── DLPFC_reference_model/
├── hippo_LIBD_reference_model/
├── hippo_SCIENCE_reference_model/
├── caudate_NCOMM_reference_model/
└── caudate_SCIENCE_reference_model/
```

## Known Issues & Exclusions

The following samples have been excluded due to quality issues:
- **DLPFC**: Br1410_DLPFC_D8, Br1442_DLPFC_E12, Br1342_DLPFC_F2, Br1193_DLPFC_B2, Br0982
- **Caudate**: Br1435_caudate_C1, Br0846_caudate_E12, Br1275_hippo_C2 (mislabeled)
- **Hippocampus**: Br1275_hippo_C2, Br1324_Hippo_A3, Br5253_Hippo_D8, Br0991_Hippo_B5

## Output Files

| File | Description |
|------|-------------|
| `{Region}_adata_consensus_full.h5ad` | Final processed dataset with annotations |
| `{Region}_harmony.h5ad` | Harmony-integrated data |
| `{Region}_scvi_top200/adata.h5ad` | scVI representation with marker genes |
| `{Region}.txt` | List of processed sample files |
| `integrated.pptx` | PowerPoint presentation with visualizations |

## Performance Notes

- **Memory**: Requires GPU for scVI training (tested on NVIDIA A100)
- **GPU Device**: Specified as `cuda:1` in SCVI model training
- **Processing Time**: ~200 epochs scVI training = 1-2 hours per dataset
- **Data Size**: DLPFC dataset contains ~400k cells across 60+ samples

## Google Drive Integration

Results can be automatically uploaded to Google Drive:

```python
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LoadClientConfigFile("path/to/client_secret.json")
drive = GoogleDrive(gauth)

# Upload file
meta = {
    "title": "results.pptx",
    "parents": [{"id": "folder_id"}],
    "mimeType": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
}
f = drive.CreateFile(meta)
f.SetContentFile("results.pptx")
f.Upload()
```

## Citation

If you use this project in your research, please cite:

```bibtex
@software{kim2024aanri,
  author = {Kim, Dongsan},
  title = {AANRI: Automated Annotation and Neuronal Relationship Investigation},
  year = {2024},
  url = {https://github.com/dkim2197/AANRI_project}
}
```

## License

[Specify your license - MIT, Apache 2.0, etc.]

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---

**Last Updated**: 2026-04-23
