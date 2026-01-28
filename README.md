# Galaxy Cluster Data Analysis Pipeline

This repository contains two modular Python pipeline for analyzing spectroscopic, photometric, and x-ray data from galaxy clusters. It was designed for use in the X-SORTER (X-ray Survey Of meRging clusTErs in Redmapper) project, but, with the exception of some specific figures and LaTeX tables, is broadly applicable. 

---

## Overview

Each pipeline may be executed separately from the root directory via run_spec_phot_pipeline.py and run_xray_pipeline.py, or collectively via run_cluster_pipeline.py. Individual scripts may be run separately to accomplish specific data analysis. Because scripts rely on upstream data products, it is recommended to execute the pipelines in order (spec-phot, then x-ray) or to utilize the comprehensive driver. After initial processing, if changes are to be made (e.g. different cluster redshift range, fov, etc.), all scripts should be executed from the beginning.

---

## Notes

The X-ray reduction pipeline is in active development, while the spec/phot pipeline is a completed stand-alone pipeline. 

---
    
## Requirements

The pipeline requires Python 3.10 or newer and standard Python packages:
* numpy
* pandas
* matplotlib
* astropy
* astroquery
* scipy
* sklearn
* requests

Queries will require CASJobs Username and Password.