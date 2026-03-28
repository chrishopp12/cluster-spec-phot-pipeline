#!/usr/bin/env python3
"""
my_utils.py — COMPATIBILITY SHIM (v2 migration)

All functions have moved to cluster_pipeline.* submodules.
This file re-exports them so existing v1 code continues to work.
Remove this file in Phase 5 once all imports are updated.
"""

# Constants
from cluster_pipeline.constants import COSMO  # noqa: F401

# Generic utilities
from cluster_pipeline.utils import (  # noqa: F401
    string_to_numeric,
    find_first_val,
    to_float_or_none,
    coerce_to_numeric,
    make_directories,
    read_json,
    pop_prefixed_kwargs,
    str2bool,
    get_color_mag_functions,
    split_members_by_spec,
)

# Coordinate helpers
from cluster_pipeline.utils.coordinates import (  # noqa: F401
    make_skycoord,
    skycoord_from_df,
    get_skycoord,
    angular_sep,
    arcsec_to_pixel_std,
)

# Cosmology
from cluster_pipeline.utils.cosmology import (  # noqa: F401
    redshift_to_comoving_distance,
    redshift_to_proper_distance,
)

# Resolvers (NED/SIMBAD/redMaPPer)
from cluster_pipeline.utils.resolvers import (  # noqa: F401
    simbad_coord_lookup,
    get_name,
    get_coordinates,
    get_redshift,
    query_redmapper,
    get_redmapper_bcg_candidates,
    get_redmapper_cluster_info,
)

# Catalog I/O
from cluster_pipeline.io.catalogs import (  # noqa: F401
    get_redseq_filename,
    load_dataframes,
    load_bcg_catalog,
    read_bcg_csv,
    select_bcgs,
    bcg_basic_info,
    load_photo_coords,
)

# Image I/O
from cluster_pipeline.io.images import (  # noqa: F401
    get_optical_image,
    is_fits_valid,
)

# LaTeX output
from cluster_pipeline.io.tables import emit_latex  # noqa: F401

# Image processing
from cluster_pipeline.xray.image import (  # noqa: F401
    fill_holes,
    smoothing,
)

# Plotting utilities
from cluster_pipeline.plotting.common import (  # noqa: F401
    finalize_figure,
    resolve_save_path,
    add_scalebar,
    overlay_bcg_markers,
)

# Optical plotting + contours
from cluster_pipeline.plotting.optical import (  # noqa: F401
    add_xray_contours,
    add_density_contours,
    define_contours_fov,
    plot_optical,
)

# Deliberately omitted (removed in v2):
#   get_cluster_id — XMM OBS_ID mapping, no longer needed
