# Galaxy Cluster Xray Data Analysis Pipeline

This repository contains a modular Python pipeline for analyzing x-ray data associated with galaxy clusters. The pipeline separates subcluster members by redshift, performs GMM fitting, plots x-ray density, Gaussian gradient magnitude, and unsharp masks. Subclustering analysis subdivides field into regions and identifies subcluster members spatially. It was designed for use in the X-SORTER (X-ray Survey Of meRging clusTErs in Redmapper) project, but, with the exception of some specific figures and LaTeX tables, is broadly applicable. 

---

The pipeline:
    
* 

---

## Overview

Each script may be run individually or the entire pipeline can be run from the main driver, run_xray_pipeline.py. The entire pipeline, including spectroscopic and photometric analysis can be run via run_cluster_pipeline.py. Because scripts rely on upstream data products, it is recommended to execute the scripts in order. After initial processing, if changes are to be made (e.g. different cluster redshift range, fov, etc.), all scripts should be executed from the beginning.

In order of intended execution:

1. 
    ```bash
    process_redshifts.py
    ```

---

## Notes

The X-ray reduction pipeline is in active development.

---
    
## Requirements



