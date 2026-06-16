"""
constants.py

Single Source of Truth for Shared Defaults
---------------------------------------------------------

Holds defaults and constants referenced by multiple modules, or that
resolve inconsistencies between files. Module-specific defaults (query
timeouts, tolerances, etc.) stay in their respective modules.
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
DEFAULT_PSF_ARCSEC = 8.0  # PSF smoothing in arcsec

# ------------------------------------
# Contours & Density
# ------------------------------------
DEFAULT_CONTOUR_LEVELS = (0.5, 0, 12)  # (n_std_bottom, n_std_top, n_levels) for X-ray contours
DEFAULT_PHOT_LEVELS = 12      # Number of photometric density contour levels
DEFAULT_PHOT_SKIP = 2         # Number of lowest density levels to skip
DEFAULT_BANDWIDTH = 0.1       # KDE bandwidth for galaxy density
DEFAULT_KDE_GRID_SIZE = 300   # Grid resolution (pixels per side) for KDE density evaluation

# ------------------------------------
# Catalog Query
# ------------------------------------
DEFAULT_SEARCH_RADIUS_ARCMIN = 10.0  # Search radius for spectroscopic/photometric queries
DEFAULT_Z_PAD = 0.15  # Padding around cluster redshift for z_min/z_max
DEFAULT_SIMBAD_TIMEOUT = 10  # SIMBAD query timeout (seconds)
DEFAULT_MATCH_TOL_ARCSEC = 3.0  # Cross-match tolerance for spec<->phot and BCG matching (arcsec)

# ------------------------------------
# Red Sequence
# ------------------------------------
DEFAULT_MAG_MIN = 16.0       # Bright-end magnitude cut for red sequence fitting
DEFAULT_COLOR_BAND = 0.15    # Half-width of red sequence color band (mag)
DEFAULT_SIGMA_CLIP = 2.0     # Sigma-clip threshold for the iterative red-sequence line fit
DEFAULT_FIT_MAX_ITER = 10    # Max iterations for the iterative red-sequence line fit

# ------------------------------------
# GMM Fitting
# ------------------------------------
DEFAULT_GMM_MAX_COMPONENTS = 8     # Maximum Gaussian components for BIC selection
DEFAULT_GMM_MIN_GALAXIES = 8      # Minimum galaxies per component to keep
DEFAULT_GMM_BROAD_THRESHOLD = 0.05 # Max sigma to keep a component (field); cluster uses 0.02
GAUSSIAN_SIGMA_MULTIPLIER = 3      # +/- n*sigma window for per-Gaussian fit bounds

# ------------------------------------
# Anderson-Darling Normality Test
# ------------------------------------
DEFAULT_AD_MC_SAMPLES = 9999   # Monte-Carlo resamples for the AD-normality p-value (scipy goodness_of_fit)
DEFAULT_AD_SEED = 1234         # Fixed RNG seed so the Monte-Carlo AD p-value is reproducible run-to-run

# ------------------------------------
# Imaging
# ------------------------------------
DEFAULT_IMAGE_PIXELS = (802, 800)  # Width, height for HiPS optical image queries

# Prioritized HiPS surveys for optical cutout queries (first valid image wins)
HIPS_SURVEYS = [
    {"name": "PanSTARRS", "hips": "CDS/P/PanSTARRS/DR1/color-i-r-g"},
    {"name": "Legacy Survey", "hips": "CDS/P/DESI-Legacy-Surveys/DR10/color"},
]

# ------------------------------------
# Subcluster Defaults
# ------------------------------------
DEFAULT_RADIUS_ARCMIN = 2.5  # Subcluster member-assignment radius in arcmin

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
DEFAULT_DPI = 450                  # Figure save resolution (dots per inch)
DEFAULT_CMAP_XRAY = "magma"        # Colormap for X-ray surface-brightness images
DEFAULT_CMAP_DENSITY = "inferno"   # Colormap for galaxy-density contour fills
DEFAULT_MARKER_SIZE_BCG = 200      # Scatter marker size for BCG overlays

# ------------------------------------
# Pipeline
# ------------------------------------
PIPELINE_STAGES = ("spec", "phot", "matching", "redseq", "subclusters", "xray")

# ------------------------------------
# Coordinates
# ------------------------------------
COORDINATE_FRAME = "icrs"  # Default sky frame for SkyCoord construction
FWHM_TO_SIGMA = 2.355      # Gaussian FWHM -> sigma divisor (2*sqrt(2*ln2))

# ------------------------------------
# Standard Catalog Column Names
# ------------------------------------
COL_RA = "RA"
COL_DEC = "Dec"
COL_Z = "z"
COL_SIGMA_Z = "sigma_z"
COL_SPEC_SOURCE = "spec_source"
COL_PHOT_SOURCE = "phot_source"
