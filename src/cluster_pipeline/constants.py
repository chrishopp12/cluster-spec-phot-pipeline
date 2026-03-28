"""
Single source of truth for shared defaults and constants.

Module-specific defaults (query timeouts, tolerances, etc.) stay in their
respective modules. This file holds values referenced by multiple modules
or that resolve inconsistencies between files.
"""

from astropy.cosmology import FlatLambdaCDM

# ------------------------------------
# Cosmology
# ------------------------------------
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

# ------------------------------------
# Field of View & Imaging
# ------------------------------------
DEFAULT_FOV_ARCMIN = 6.0         # Zoomed FOV for subcluster-scale images
DEFAULT_FOV_FULL_ARCMIN = 30.0   # Full FOV for wide-field images
DEFAULT_RA_OFFSET_ARCMIN = 0.0   # Image center offset in RA
DEFAULT_DEC_OFFSET_ARCMIN = 0.0  # Image center offset in Dec
DEFAULT_SURVEY = "legacy"        # Photometry survey: "legacy" or "panstarrs"
DEFAULT_COLOR_TYPE = "gr"        # Color band for red sequence: "gr", "ri", "gi"

# ------------------------------------
# X-ray
# ------------------------------------
DEFAULT_XRAY_FILENAME = "comb-adaptimsky-400-1100.fits"  # Standard XMM combined image
DEFAULT_PSF_ARCSEC = 8.0  # PSF smoothing in arcsec (was 8.0 in cluster.py, 10.0 in run_xray_pipeline.py — using 8.0)

# ------------------------------------
# Contours & Density
# ------------------------------------
DEFAULT_CONTOUR_LEVELS = (0.5, 0, 12)  # (n_std_bottom, n_std_top, n_levels) for X-ray contours
DEFAULT_PHOT_LEVELS = 12      # Number of photometric density contour levels
DEFAULT_PHOT_SKIP = 2         # Number of lowest density levels to skip
DEFAULT_BANDWIDTH = 0.1       # KDE bandwidth for galaxy density

# ------------------------------------
# Redshift
# ------------------------------------
DEFAULT_Z_PAD = 0.15  # Padding around cluster redshift for z_min/z_max

# ------------------------------------
# Subcluster Defaults
# ------------------------------------
DEFAULT_RADIUS_MPC = 2.5  # Subcluster search radius in Mpc

DEFAULT_COLORS = [
    "white", "tab:green", "tab:purple", "tab:cyan", "tab:pink",
    "gold", "tab:orange", "tab:green", "tab:purple", "tab:pink",
]

DEFAULT_LABELS = [
    "1", "2", "3", "4", "5",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
]

# ------------------------------------
# Plotting
# ------------------------------------
DEFAULT_LEGEND_LOC = "upper right"
DEFAULT_SAVE_PLOTS = True
DEFAULT_SHOW_PLOTS = False
