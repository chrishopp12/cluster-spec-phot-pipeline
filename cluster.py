#!/usr/bin/env python3
"""
cluster.py

Cluster Class Definition
---------------------------------------------------------

Defines the Cluster class for managing galaxy cluster data, including
metadata, file paths, and data retrieval.

This class centralizes:
- Cluster metadata (ID, name, coordinates, redshift, richness, etc.)
- File path management for cluster-specific data (photometry, redshifts, X-ray, images)
- Data population from CSV files, external queries, and defaults
- Retrieval of optical images from surveys with caching.

Notes:
- populate() method fills in missing data in order of precedence:
  1. CLI/init values
  2. clusters.csv
  3. External queries (NED/SIMBAD)
  4. Default values
"""
from __future__ import annotations

import os
import re
import csv

import pandas as pd

from pathlib import Path
from typing import Any
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

from astroquery.hips2fits import hips2fits

from my_utils import get_name, get_coordinates, get_redshift, get_redseq_filename, is_fits_valid


# ------------------------------------
# Helpers
# ------------------------------------

def _is_missing(val: Any) -> bool:
    """Check if a value is considered 'missing' (None, empty strings, NaN).

    Parameters
    ----------
    val : Any
        The value to check.

    Returns
    -------
    bool
        True if the value is missing, False otherwise.  
     """

    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    try:
        if bool(pd.isna(val)):
            return True
    except Exception:
        pass
    return False

def _safe_float(val: Any) -> float | None:
    if _is_missing(val):
        return None
    try:
        return float(val)
    except Exception:
        return None
    
def _safe_int(val: Any) -> int | None:
    if _is_missing(val):
        return None
    try:
        return int(val)
    except Exception:
        return None

    
def _clean_folder_name(identifier: str) -> str:
    """Make a stable folder name from a cluster identifier."""
    return "_".join(str(identifier).strip().split())


class Cluster:

# ------------------------------------
# Defaults/ Constants
# ------------------------------------
    BASE_PATH_ENV_VAR = "XSORTER_CLUSTER_BASE_PATH"
    _REPO_ROOT = Path(__file__).resolve().parent
    _REPO_CLUSTER_PATH = _REPO_ROOT.parent/"Clusters"
    DEFAULT_BASE_PATH = Path(os.environ.get(BASE_PATH_ENV_VAR, _REPO_CLUSTER_PATH)).expanduser().resolve()
    MASTER_FILE_NAME = "clusters.csv"
    BCG_FILE_NAME = "candidate_mergers.csv"

    DEFAULT_FOV_ARCMIN = 6.0
    DEFAULT_FOV_FULL_ARCMIN = 30.0
    DEFAULT_RA_OFFSET_ARCMIN = 0.0
    DEFAULT_DEC_OFFSET_ARCMIN = 0.0

    DEFAULT_SURVEY = "legacy"
    DEFAULT_COLOR_TYPE = "gr"

    DEFAULT_PHOT_LEVELS = 12      
    DEFAULT_PHOT_SKIP = 2
    DEFAULT_CONTOUR_LEVELS = "0.5,0,12"
    DEFAULT_PSF = 8.0 
    DEFAULT_BANDWIDTH = 0.1

    CSV_COLUMNS = (
        'identifier',
        'name',
        'ra',
        'dec',
        'ra_offset',
        'dec_offset',
        'redshift',
        'redshift_err',
        'z_min',
        'z_max',
        'richness',
        'richness_err',
        'fov',
        'fov_full',
        'phot_levels',
        'phot_skip',
        'contour_levels',
        'psf',
        'bandwidth',
        'survey',
        'color_type',
    )


# ------------------------------------
# Init/ Representation
# ------------------------------------

    def __init__(
        self,
        identifier: str,
        *,
        base_path: str | None = None,
        master_csv_file: str | None = None,
        bcg_csv_file: str | None = None,
        **kwargs,
    ):

        self.identifier = str(identifier).strip()

        self.base_path = str(base_path).strip() if base_path else self.DEFAULT_BASE_PATH
        self.master_csv_file = (
            str(master_csv_file).strip()
            if master_csv_file else os.path.join(self.base_path, self.MASTER_FILE_NAME)
        )
        self.bcg_csv_file = (
            str(bcg_csv_file).strip()
            if bcg_csv_file else os.path.join(self.base_path, self.BCG_FILE_NAME)
        )

        # Persistent fields
        self.name: str | None = kwargs.get('name')
        self.ra: float | None = _safe_float(kwargs.get('ra'))
        self.dec: float | None = _safe_float(kwargs.get('dec'))
        self.ra_offset: float | None = _safe_float(kwargs.get('ra_offset'))
        self.dec_offset: float | None = _safe_float(kwargs.get('dec_offset'))
        self.redshift: float | None = _safe_float(kwargs.get('redshift'))
        self.redshift_err: float | None = _safe_float(kwargs.get('redshift_err'))
        self.z_min: float | None = _safe_float(kwargs.get('z_min'))
        self.z_max: float | None = _safe_float(kwargs.get('z_max'))
        self.richness: float | None = _safe_float(kwargs.get('richness'))
        self.richness_err: float | None = _safe_float(kwargs.get('richness_err'))
        self.fov: float | None = _safe_float(kwargs.get('fov'))
        self.fov_full: float | None = _safe_float(kwargs.get('fov_full'))
        self.phot_levels: int | None = _safe_int(kwargs.get('phot_levels'))
        self.phot_skip: int | None = _safe_int(kwargs.get('phot_skip'))
        self.contour_levels: str | tuple[float, float, float] | list[float] | None = kwargs.get('contour_levels')
        self.psf: float | None = _safe_float(kwargs.get('psf'))
        self.bandwidth: float | None = _safe_float(kwargs.get('bandwidth'))
        self.survey: str | None = kwargs.get('survey')
        self.color_type: str | None = kwargs.get('color_type')

        self._coords: SkyCoord | None = None
        self._init_paths()


    def __repr__(self) -> str:

        ra = f"{self.ra:.6f}" if isinstance(self.ra, (int, float)) else "None"
        dec = f"{self.dec:.6f}" if isinstance(self.dec, (int, float)) else "None"

        output =  "\n#######################################\n"
        output +=  f"        Cluster: {self.identifier} \n"
        output += f"           Name: {self.name}\n"
        output += f"       Redshift: {self.redshift}\n"
        output += f"          z_min: {self.z_min}\n"
        output += f"          z_max: {self.z_max}\n"
        output += f"    Coordinates: ({ra}, {dec})\n"
        output += "#######################################\n"
        return output


# ------------------------------------
# Public Methods
# ------------------------------------

    def populate(
            self,
            *,
            update_csv: bool = True,
            verbose: bool = False
    ) -> None:
        """
        Fill all cluster fields in order:
            1. CLI/init (already set in __init__)
            2. clusters.csv (calls self.populate_from_csv)
            3. fetch (calls self.populate_from_fetch)
            4. defaults (calls self.populate_from_defaults)
        Updates clusters.csv if 'update_csv' is True.
        """
        if verbose:
            print(f"[Cluster] Populating {self.identifier} ...")

        self._populate_from_csv(verbose=verbose)
        self._populate_from_fetch(verbose=verbose)
        self._populate_from_defaults(verbose=verbose)

        self._init_paths()

        if update_csv:
            self._update_csv()
        if verbose:
            print(f"[Cluster] Populated and updated CSV for {self.identifier}")


    def set_params(self, **kwargs: Any) -> None:
        """
        Update multiple cluster attributes and sync to CSV.
        Example: cluster.set_params(phot_levels=10, psf=7.5)
        """
        updated = False
        for key, val in kwargs.items():
            if not hasattr(self, key):
                continue
            original_val = getattr(self, key)
            if original_val != val:
                setattr(self, key, val)
                updated = True
        if updated:
            # Normalize numeric fields
            self.ra = _safe_float(self.ra)
            self.dec = _safe_float(self.dec)
            self.ra_offset = _safe_float(self.ra_offset)
            self.dec_offset = _safe_float(self.dec_offset)
            self.redshift = _safe_float(self.redshift)
            self.redshift_err = _safe_float(self.redshift_err)
            self.z_min = _safe_float(self.z_min)
            self.z_max = _safe_float(self.z_max)
            self.richness = _safe_float(self.richness)
            self.richness_err = _safe_float(self.richness_err)
            self.fov = _safe_float(self.fov)
            self.fov_full = _safe_float(self.fov_full)
            self.psf = _safe_float(self.psf)
            self.bandwidth = _safe_float(self.bandwidth)

            self._coords = None
            self._init_paths()
            self._update_csv()


    def get_phot_file(
            self,
            survey: str | None = None,
            color_type: str | None = None
        ) -> str:
        """
        Return the photometric catalog path for the requested survey and color_type.
        If not specified, uses the object's defaults.
        """
        survey_eff = survey or self.survey
        color_type_eff = color_type or self.color_type
        if survey_eff is None or color_type_eff is None:
            raise ValueError("Survey/ color_type not set. Run cluster.populate() or specify them in the call.")
        return get_redseq_filename(self.photometry_path, survey_eff, color_type_eff)


    def get_optical_image(
            self,
            *,
            fov: float | None = None,
            ra_offset: float | None = None,
            dec_offset: float | None = None,
            check_validity: bool = True,
            verbose: bool = False
        ) -> str | None:
        """
        Retrieve an optical image for this cluster, using cached file or querying surveys.

        Parameters
        ----------
        fov : float | None
            Field of view in arcminutes. If None, uses instance default.
        ra_offset : float | None
            RA offset in arcminutes. If None, uses instance default.
        dec_offset : float | None
            Dec offset in arcminutes. If None, uses instance default.
        check_validity : bool
            If True, checks if retrieved FITS file is valid (non-empty).
        verbose : bool
            If True, prints progress messages.

        Returns
        -------
        str | None
            Path to the retrieved FITS file, or None if no data available.

        Notes
        -----
        Prioritizes Legacy Survey, then PanSTARRS.
        """
        # Use instance defaults unless overridden
        fov_eff = float(fov) if fov is not None else float(self.fov or self.DEFAULT_FOV_ARCMIN)
        ra_offset_eff = float(ra_offset) if ra_offset is not None else float(self.ra_offset or self.DEFAULT_RA_OFFSET_ARCMIN)
        dec_offset_eff = float(dec_offset) if dec_offset is not None else float(self.dec_offset or self.DEFAULT_DEC_OFFSET_ARCMIN)

        if self.coords is None:
            raise ValueError("Cluster coordinates (RA/Dec) are not set. Cannot retrieve optical image.")
        
        # Convert arcminutes to degrees for query
        ra_offset_deg = ra_offset_eff / 60
        dec_offset_deg = dec_offset_eff / 60
        fov_deg = fov_eff / 60

        # Define survey priority list
        surveys = [
            {'name': 'Legacy Survey', 'hips': 'CDS/P/DESI-Legacy-Surveys/DR10/color'},
            {'name': 'PanSTARRS', 'hips': 'CDS/P/PanSTARRS/DR1/color-i-r-g'},
        ]

        # ----- File search logic -----
        # 1. Check for cached file variants
        for fv, rv, dv in self._file_variants(fov_eff, ra_offset_eff, dec_offset_eff):
            fits_path = os.path.join(self.photometry_path, f"optical_image_{fv}_{rv}_{dv}.fits")
            if is_fits_valid(fits_path):
                if verbose:
                    print(f"Found cached optical image: {fits_path}")
                return fits_path

        # 2. Query surveys in order of priority
        for survey in surveys:
            try:
                print(f"Querying {survey['name']}...")
                result = hips2fits.query(
                    hips=survey['hips'],
                    width=802,
                    height=800,
                    ra=self.coords.ra + ra_offset_deg * u.deg,
                    dec=self.coords.dec + dec_offset_deg * u.deg,
                    fov=Angle(fov_deg, 'deg'),
                    projection='TAN',
                    format='fits'
                )
                survey_path = os.path.join(self.photometry_path, f"{survey['name']}_optical_image_{fov_eff}_{ra_offset_eff}_{dec_offset_eff}.fits")
                fits_path = os.path.join(self.photometry_path, f"optical_image_{fov_eff}_{ra_offset_eff}_{dec_offset_eff}.fits")
                result.writeto(survey_path, overwrite=True)
                result.writeto(fits_path, overwrite=True)
                print(f"Image retrieved and saved: {survey_path}")

                if check_validity and not is_fits_valid(survey_path):
                    print(f"{survey['name']} image appears empty, checking the next survey...")
                    try:
                        os.remove(survey_path)
                    except OSError:
                        pass
                    try:
                        os.remove(fits_path)
                    except OSError:
                        pass
                    continue
                return fits_path
            except Exception as e:
                print(f"Failed to retrieve image from {survey['name']}: {e}")

        print("No data available from the prioritized surveys.")
        return None


# ------------------------------------
# Private Methods
# ------------------------------------

    def _init_paths(self) -> None:
        folder_name = _clean_folder_name(self.identifier)

        self._cluster_path = os.path.join(self.base_path, folder_name)
        self._redshift_path = os.path.join(self._cluster_path, "Redshifts")
        self._photometry_path = os.path.join(self._cluster_path, "Photometry")
        self._xray_path = os.path.join(self._cluster_path, "Xray")
        self._image_path = os.path.join(self._cluster_path, "Images")

        self._xray_file = os.path.join(self._xray_path, "comb-adaptimsky-400-1100.fits")
        self._spec_file = os.path.join(self._redshift_path, "combined_redshifts.csv")
        self._bcg_file = os.path.join(self._redshift_path, "BCGs.csv")
        self._subcluster_file = os.path.join(self._cluster_path, "subclusters.csv")


    def _populate_from_csv(self, *, verbose: bool = True) -> None:
        """Fill missing fields from clusters.csv."""
        if not os.path.exists(self.master_csv_file):
            return
        
        df = pd.read_csv(self.master_csv_file)
        if 'identifier' not in df.columns:
            return

        row = df.loc[df['identifier'].astype(str) == str(self.identifier)]
        if row.empty:
            return
        
        csv_row = row.iloc[0].to_dict()
        for column in self.CSV_COLUMNS:
            if column == 'identifier':
                continue
            if not hasattr(self, column):
                continue
            current_val = getattr(self, column)
            if _is_missing(current_val) and column in csv_row:
                setattr(self, column, csv_row[column])
                if verbose:
                    print(f"  [from CSV] {column}: {getattr(self, column)}")

        # Normalize numeric fields
        self.ra = _safe_float(self.ra)
        self.dec = _safe_float(self.dec)
        self.ra_offset = _safe_float(self.ra_offset)
        self.dec_offset = _safe_float(self.dec_offset)
        self.redshift = _safe_float(self.redshift)
        self.redshift_err = _safe_float(self.redshift_err)
        self.z_min = _safe_float(self.z_min)
        self.z_max = _safe_float(self.z_max)
        self.richness = _safe_float(self.richness)
        self.richness_err = _safe_float(self.richness_err)
        self.fov = _safe_float(self.fov)
        self.fov_full = _safe_float(self.fov_full)
        self.psf = _safe_float(self.psf)
        self.bandwidth = _safe_float(self.bandwidth)
        self.phot_levels = _safe_int(self.phot_levels)
        self.phot_skip = _safe_int(self.phot_skip)


    def _populate_from_fetch(self, *, verbose: bool = True) -> None:
        """Fill remaining fields by querying external sources."""

        # Name/Coords
        if _is_missing(self.ra) or _is_missing(self.dec) or _is_missing(self.name):
            self.name = self.name or get_name(self.identifier)
            coords = get_coordinates(self.identifier)
            self.ra, self.dec = float(coords.ra.deg), float(coords.dec.deg)
            self._coords = coords
            if verbose:
                print(f"  [fetch] name/coords: {self.name}, {self.ra}, {self.dec}")
        else:
            self._coords = SkyCoord(ra=float(self.ra) * u.deg, dec=float(self.dec) * u.deg)

        # Redshift/Richness (prefer candidate_mergers.csv)
        needs_z = _is_missing(self.redshift)
        needs_richness = _is_missing(self.richness)
        
        if needs_z or needs_richness:
            from_csv = False
            if os.path.exists(self.bcg_csv_file) and not _is_missing(self.name):
                try:
                    df_bcg = pd.read_csv(self.bcg_csv_file)
                    if "NAME" in df_bcg.columns:
                        row = df_bcg.loc[df_bcg["NAME"].astype(str) == str(self.name)]
                        if not row.empty:
                            row_0 = row.iloc[0]
                            self.redshift = _safe_float(row_0.get("Z_LAMBDA"))
                            self.redshift_err = _safe_float(row_0.get("Z_LAMBDA_ERR"))
                            self.richness = _safe_float(row_0.get("LAMBDA"))
                            self.richness_err = _safe_float(row_0.get("LAMBDA_ERR"))
                            from_csv = True
                            if verbose:
                                print(f"  [fetch] redshift/richness from candidate_mergers: {self.redshift}, {self.richness}")

                except Exception as e:
                    print(f"[WARNING] candidate_mergers lookup failed for {self.identifier}: {e}")

            if not from_csv and _is_missing(self.redshift):
                # Fallback to external query (NED/ SIMBAD)
                self.redshift = _safe_float(get_redshift(self.identifier))
                self.redshift_err = None
                if verbose:
                    print(f"  [fetch] redshift from external query: {self.redshift}")


    def _populate_from_defaults(self, *, verbose: bool = True) -> None:
        """Set any remaining fields to defaults."""
        defaults: dict[str, Any] = {
            'fov': self.DEFAULT_FOV_ARCMIN,
            'fov_full': self.DEFAULT_FOV_FULL_ARCMIN,
            'ra_offset': self.DEFAULT_RA_OFFSET_ARCMIN,
            'dec_offset': self.DEFAULT_DEC_OFFSET_ARCMIN,
            'survey': self.DEFAULT_SURVEY,
            'color_type': self.DEFAULT_COLOR_TYPE,
            'phot_levels': self.DEFAULT_PHOT_LEVELS,
            'phot_skip': self.DEFAULT_PHOT_SKIP,
            'contour_levels': self.DEFAULT_CONTOUR_LEVELS,
            'psf': self.DEFAULT_PSF,
            'bandwidth': self.DEFAULT_BANDWIDTH,
        }

        for key, default in defaults.items():
            if _is_missing(getattr(self, key, None)):
                setattr(self, key, default)
                if verbose:
                    print(f"  [default] {key}: {default}")


    def _update_csv(self) -> None:
        """
        Update (or create) this cluster's row in MASTER_CSV.
        Expands columns if necessary.
        """
        row: dict[str, Any] = {c: None for c in self.CSV_COLUMNS}
        row["identifier"] = self.identifier
        row["name"] = self.name
        row["ra"] = self.ra
        row["dec"] = self.dec
        row["ra_offset"] = self.ra_offset
        row["dec_offset"] = self.dec_offset
        row["redshift"] = self.redshift
        row["redshift_err"] = self.redshift_err
        row["z_min"] = self.z_min
        row["z_max"] = self.z_max
        row["richness"] = self.richness
        row["richness_err"] = self.richness_err
        row["fov"] = self.fov
        row["fov_full"] = self.fov_full
        row["phot_levels"] = self.phot_levels
        row["phot_skip"] = self.phot_skip
        row["contour_levels"] = self.contour_levels
        row["psf"] = self.psf
        row["bandwidth"] = self.bandwidth
        row["survey"] = self.survey
        row["color_type"] = self.color_type
        

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.master_csv_file), exist_ok=True)

        existing_rows: list[dict[str, Any]] = []
        existing_fieldnames: list[str] = []

        if os.path.exists(self.master_csv_file):
            with open(self.master_csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_fieldnames = list(reader.fieldnames or [])
                for r in reader:
                    existing_rows.append(r)

        # Preserve any extra columns that already exist in the file
        extra_cols = [c for c in existing_fieldnames if c not in self.CSV_COLUMNS]
        fieldnames = list(self.CSV_COLUMNS) + extra_cols

        # Remove any existing row for this identifier
        ident_str = str(self.identifier)
        filtered_rows: list[dict[str, Any]] = []
        for r in existing_rows:
            if str(r.get("identifier", "")).strip() == ident_str:
                continue
            filtered_rows.append(r)

        # Construct the new row matching the final header (include extras as blanks)
        out_row = {c: row.get(c, None) for c in fieldnames}

        # Write back (header + filtered + new)
        with open(self.master_csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in filtered_rows:
                # Make sure every written row has all columns (missing -> blank)
                writer.writerow({c: r.get(c, "") for c in fieldnames})
            writer.writerow(out_row)



    @staticmethod
    def _file_variants(
        fov: float,
        ra_offset: float,
        dec_offset: float
    ) -> list[tuple[str, str, str]]:
        """Generate possible string variants for file naming."""
        fov_float = float(fov)
        ra_float = float(ra_offset)
        dec_float = float(dec_offset)

        fov_int = int(round(fov_float))
        ra_int = int(round(ra_float))
        dec_int = int(round(dec_float))
        variants: list[tuple[str, str, str]] = [
            (str(fov_float), str(ra_float), str(dec_float)),
            (str(fov_int), str(ra_int), str(dec_int)),
            (f"{fov_float:.1f}", f"{ra_float:.1f}", f"{dec_float:.1f}")
        ]
        # Ensure uniqueness and preserve order
        seen = set()
        unique_variants = []
        for a, b, c in variants:
            key = (a, b, c)
            if key not in seen:
                seen.add(key)
                unique_variants.append(key)
        return unique_variants



# ------------------------------------
# Properties
# ------------------------------------
    @property
    def cluster_path(self) -> str:
        return self._cluster_path

    @property
    def redshift_path(self) -> str:
        return self._redshift_path

    @property
    def photometry_path(self) -> str:
        return self._photometry_path
    
    @property
    def xray_path(self) -> str:
        return self._xray_path

    @property
    def image_path(self) -> str:
        return self._image_path

    @property
    def xray_file(self) -> str:
        return self._xray_file

    @property
    def spec_file(self) -> str:
        return self._spec_file

    @property
    def bcg_file(self) -> str:
        return self._bcg_file

    @property
    def subcluster_file(self) -> str:
        return self._subcluster_file

    @property
    def coords(self) -> SkyCoord | None:
        """
        Returns SkyCoord object for cluster RA/Dec, if available.
        """

        if self._coords is not None:
            return self._coords
        
        if _is_missing(self.ra) or _is_missing(self.dec):
            return None
        
        try:
            self._coords = SkyCoord(ra=float(self.ra) * u.deg, dec=float(self.dec) * u.deg)
        except Exception:
            self._coords = None

        return self._coords

    @property
    def coords_str(self) -> str:
        if self.coords is None:
            return "RA: None, Dec: None"
        return f"RA: {self.coords.ra.deg:.6f}, Dec: {self.coords.dec.deg:.6f}"
    
    @property
    def contour_levels_tuple(self) -> tuple[float, float, float]:
        """
        Returns (n_std_from_bottom, n_std_from_top, n_levels) as floats.
        Accepts:
            - tuple/list of three floats
            - comma-separated string "0.5,0,12"
            - space-separated string "0.5 0 12"
        """
        val = self.contour_levels

        if isinstance(val, (tuple, list)):
            if len(val) != 3:
                raise ValueError(f"contour_levels must have three values. Got: {val}")
            return float(val[0]), float(val[1]), float(val[2])

        # Remove parentheses and extra spaces
        val_str = str(val).strip().replace("(", "").replace(")", "")
        # Split on comma or whitespace
        parts = [p for p in re.split(r'[,\s]+', val_str) if p]
        if len(parts) != 3:
            raise ValueError(f"contour_levels must have three values separated by comma or space. Got: {parts}")
        return float(parts[0]), float(parts[1]), float(parts[2])


