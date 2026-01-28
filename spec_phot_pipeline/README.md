# Galaxy Cluster Spectroscopy and Photometry Data Analysis Pipeline

This repository contains a modular Python pipeline for analyzing spectroscopic and photometric data associated with galaxy clusters. The pipeline automates retrieval of archival redshift and photometric catalogs, identifies cluster members via a fit of the red sequence, and produces summary catalogs and diagnostic plots. It was designed for use in the X-SORTER (X-ray Survey Of meRging clusTErs in Redmapper) project, but, with the exception of some specific figures and LaTeX tables, is broadly applicable. 

---

The pipeline:
    
* Queries archival spectroscopic redshifts and joins with user-supplied values.
* Queries archival photometry.
* Constructs a color-magnitude diagram and identifies photometric cluster members through a fit of the cluster red sequence.
* Builds joined spectroscopic and photometric catalogs.
* Produces luminosity-weighted galaxy density maps.

---

## Overview

Each script may be run individually or the entire pipeline can be run from the main driver, run_cmd_pipeline.py. Because scripts rely on upstream data products, it is recommended to execute the scripts in order. After initial processing, if changes are to be made (e.g. different cluster redshift range, fov, etc.), all scripts should be executed from the beginning.

In order of intended execution:

1. 
    ```bash
    archival_z_pipeline.py
    ```
    * Queries NED, SDSS, and DESI for archival redshifts.
    * Produces individual survey redshift catalogs:
        * ned.csv
        * sdss.csv
        * desi.csv
    * Combines and deduplicates catalogs:
        * archival_z.csv
        * archival_z.txt

2. 
    ```bash
    archival_phot_pipeline.py
    ```
    * Queries Legacy and PanSTARRS photometric surveys.
    * Calculates colors and luminosity weights.
    * Produces individual catalogs:
        * photometry_legacy.csv
        * photometry_PanSTARRS.csv
    * Combines results into a single catalog:
        * photometry_combined.csv

3. 
    ```bash
    make_catalogs_pipeline.py
    ```
    * Requires catalogs from previous steps:
        * archival_z.csv
        * photometry_legacy.csv
        * photometry_PanSTARRS.csv
    * Merges user-supplied spectroscopy files (zdump*.txt) and saves:
        * deimos.csv
    * Joins spectroscopic and photometric catalogs by survey:
        * survey_matched.csv : All sources with both photometric and spectroscopic data.
        * survey_catalog.csv : All sources with photometric or spectroscopic data.
    * Loads BCG data from file (candidate_mergers.csv) or redMaPPer query, matches to spectroscopy, and saves:
        * BCGs.csv

4. 
    ```bash
    color_magnitude_pipeline.py
    ```
    * Requires catalogs from previous steps:
        * survey_catalog.csv
        * survey_matched.csv
    * Constructs cluster color-magnitude diagrams for each survey and color-band.
    * Fits red sequence using spectroscopic members.
    * Identifies members and saves:
        * redseq_{survey}_{color}.csv
    * Produces diagnostic CMD plots.

5. 
    ```bash
    color_magnitude_plotting.py
    ```
    * Provides plotting routines:
        * CMDs
        * Member spatial distribution
        * Luminosity-weighted galaxy density maps

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
