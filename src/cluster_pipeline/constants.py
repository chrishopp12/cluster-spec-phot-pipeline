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
DEFAULT_RADIUS_MPC = 2.0  # Subcluster search radius in Mpc

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

# ------------------------------------
# XMM Observation ID → RMJ Name Mapping
# ------------------------------------
# Maps XMM OBS_IDs to canonical redMaPPer cluster identifiers.
# Used by get_cluster_id() to resolve short identifiers.
CLUSTER_MAP = {
    "1234567890": "RMJ213518.8+012527.0",
    "0881900801": "RMJ000343.8+100123.8",
    "0881901001": "RMJ080135.3+362807.5",
    "0881901201": "RMJ132724.2+534656.5",
    "0901870201": "RMJ092647.3+050004.0",
    "0881900301": "RMJ121917.6+505432.8",
    "0901870901": "RMJ082944.9+382754.4",
    "0881900501": "RMJ163509.2+152951.5",
    "0881900701": "RMJ125725.9+365429.4",
    "0922150101": "RMJ232104.1+291134.5",
    "0922150301": "RMJ104311.1+150151.9",
    "0922150401": "RMJ010934.2+330301.0",
    "0922150601": "RMJ021952.2+012952.2",
}
