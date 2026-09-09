# Phase 0 — Dataset Download & Setup

## 1. Download Crops3D

Direct download (PowerShell):
```powershell
Invoke-WebRequest -Uri "https://springernature.figshare.com/ndownloader/files/50027964" -OutFile "Crops3D.zip"
Expand-Archive -Path "Crops3D.zip" -DestinationPath ".\Crops3D_raw"
```

This gives you (top-level directories inside the zip):
```
Crops3D/            <- raw annotated point clouds, PLY files, 8 crop-type subfolders
Crops3D_10k/        <- same data, farthest-point-sampled to 10,000 points/cloud (use this for speed)
Crops3D_100k-C/      <- corruption-robustness test variants (ignore for Phase 0)
Crops3D_IS/          <- instance segmentation annotations (ignore for now)
```
You only need `Tomato/` and `Maize/` inside `Crops3D/` (or `Crops3D_10k/`) for this project.
Reference: Zhu et al., "Crops3D: a diverse 3D crop dataset...", Scientific Data, 2024.
DOI: 10.6084/m9.figshare.27313272

## 2. Download Pheno4D

```powershell
Invoke-WebRequest -Uri "https://www.ipb.uni-bonn.de/html/projects/Pheno4D/Pheno4D.zip" -OutFile "Pheno4D.zip"
Expand-Archive -Path "Pheno4D.zip" -DestinationPath ".\Pheno4D_raw"
```
If that direct link 404s (Uni Bonn occasionally moves it), get the current link from
https://www.ipb.uni-bonn.de/data/pheno4d/index.html and swap it in above.

Folder structure (note: the real archive ships `.txt` files, not `.xyz` as
some secondary sources describe — the loader in `data_io.py` accepts both):
```
Pheno4D/
  Maize01/  M01_0313_a.txt  M01_0314.txt  ...   (filenames ending _a = annotated)
  Maize02/  ...
  ...
  Tomato01/ T01_0305_a.txt  T01_0306.txt  ...
  ...
```
Annotated tomato file columns: x, y, z, label (4 columns)
Annotated maize file columns:  x, y, z, label_leaf_collar, label_leaf_tip (5 columns)
Unannotated files: x, y, z only (3 columns)

Reference: Schunck et al., "Pheno4D...", PLOS ONE, 2021.

## 3. Expected local layout for the scripts below

```
data/
  crops3d/
    Tomato/*.ply
    Maize/*.ply
  pheno4d/
    Maize01/*.txt  Maize02/*.txt  ... Maize07/*.txt
    Tomato01/*.txt ... Tomato07/*.txt
```
Adjust `--crops3d_root` / `--pheno4d_root` in each script if your folder names differ
(e.g. if Pheno4D unzips as `maize/plant1` lowercase instead of `Maize01` — check what
you actually got and either rename or edit `data_io.py`'s glob patterns to match).

## 4. Install dependencies

```powershell
pip install numpy pandas matplotlib scipy open3d tqdm plyfile
```
