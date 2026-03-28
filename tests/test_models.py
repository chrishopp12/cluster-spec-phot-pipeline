"""Tests for BCG, Region, and Subcluster data classes."""

import pytest
import pandas as pd

from cluster_pipeline.models.bcg import BCG
from cluster_pipeline.models.region import Region
from cluster_pipeline.models.subcluster import Subcluster


# ------------------------------------
# BCG
# ------------------------------------

class TestBCG:

    def test_basic_construction(self):
        bcg = BCG(bcg_id=1, ra=201.85, dec=53.78, z=0.441, label="1")
        assert bcg.bcg_id == 1
        assert bcg.ra == 201.85
        assert bcg.z == 0.441
        assert bcg.has_redshift is True

    def test_no_redshift(self):
        bcg = BCG(bcg_id=2, ra=201.79, dec=53.80)
        assert bcg.has_redshift is False
        assert bcg.z is None

    def test_coord_property(self):
        bcg = BCG(bcg_id=1, ra=180.0, dec=45.0)
        coord = bcg.coord
        assert abs(coord.ra.deg - 180.0) < 1e-10
        assert abs(coord.dec.deg - 45.0) < 1e-10

    def test_from_config(self):
        cfg = {"ra": 201.85, "dec": 53.78, "z": 0.441, "rank": 1, "prob": 0.95}
        bcg = BCG.from_config(bcg_id=1, cfg=cfg)
        assert bcg.bcg_id == 1
        assert bcg.probability == 0.95
        assert bcg.rank == 1
        assert bcg.label == "1"  # defaults to str(bcg_id)

    def test_from_config_with_label(self):
        cfg = {"ra": 201.0, "dec": 53.0, "label": "Main"}
        bcg = BCG.from_config(bcg_id=1, cfg=cfg)
        assert bcg.label == "Main"

    def test_to_config_round_trip(self):
        bcg = BCG(bcg_id=3, ra=201.0, dec=53.0, z=0.44, rank=2, probability=0.8, label="A")
        cfg = bcg.to_config()
        assert cfg["ra"] == 201.0
        assert cfg["z"] == 0.44
        assert cfg["prob"] == 0.8
        assert cfg["label"] == "A"

    def test_to_config_omits_none(self):
        bcg = BCG(bcg_id=1, ra=201.0, dec=53.0)
        cfg = bcg.to_config()
        assert "z" not in cfg
        assert "rank" not in cfg

    def test_from_dataframe_row(self):
        row = pd.Series({
            "RA": 201.85, "Dec": 53.78, "z": 0.441,
            "sigma_z": 0.001, "BCG_priority": 1, "BCG_probability": 0.95,
        })
        bcg = BCG.from_dataframe_row(row)
        assert bcg.bcg_id == 1
        assert bcg.z == pytest.approx(0.441)
        assert bcg.probability == pytest.approx(0.95)

    def test_repr(self):
        bcg = BCG(bcg_id=1, ra=201.85, dec=53.78, z=0.441, label="1")
        r = repr(bcg)
        assert "BCG" in r
        assert "0.4410" in r


# ------------------------------------
# Region
# ------------------------------------

class TestRegion:

    def test_basic_construction(self):
        region = Region(bcg_index=0, signature=[1, -1, 0])
        assert region.bcg_index == 0
        assert region.signature == [1, -1, 0]

    def test_matches_same_signature(self):
        region = Region(bcg_index=0, signature=[1, -1, 0])
        assert region.matches([1, -1, 0]) is True

    def test_matches_zero_wildcard(self):
        """Zero in region signature matches any sign in point."""
        region = Region(bcg_index=0, signature=[1, 0, -1])
        assert region.matches([1, 1, -1]) is True
        assert region.matches([1, -1, -1]) is True

    def test_matches_zero_in_point(self):
        """Zero in point signature matches any sign in region."""
        region = Region(bcg_index=0, signature=[1, -1])
        assert region.matches([1, 0]) is True

    def test_no_match(self):
        region = Region(bcg_index=0, signature=[1, -1])
        assert region.matches([-1, -1]) is False

    def test_repr(self):
        region = Region(bcg_index=0, signature=[1, -1, 0], segments=[(1, 2), (3, 4)])
        r = repr(region)
        assert "+-0" in r
        assert "n_segments=2" in r


# ------------------------------------
# Subcluster
# ------------------------------------

class TestSubcluster:

    @pytest.fixture
    def bcgs(self):
        """Create a set of test BCGs."""
        return {
            1: BCG(bcg_id=1, ra=201.85, dec=53.78, z=0.441, label="1"),
            2: BCG(bcg_id=2, ra=201.79, dec=53.80, z=0.438, label="2"),
            5: BCG(bcg_id=5, ra=201.82, dec=53.77, z=0.440, label="A"),
        }

    def test_basic_construction(self, bcgs):
        sub = Subcluster(
            bcg_id=1,
            primary_bcg=bcgs[1],
            label="Main",
            color="white",
            radius_mpc=2.5,
            z_range=(0.42, 0.46),
        )
        assert sub.bcg_id == 1
        assert sub.label == "Main"
        assert sub.n_spec == 0
        assert sub.n_phot == 0

    def test_region_center_defaults_to_primary(self, bcgs):
        """Without a border_bcg, region_center uses primary_bcg."""
        sub = Subcluster(bcg_id=1, primary_bcg=bcgs[1])
        assert abs(sub.region_center.ra.deg - 201.85) < 1e-10

    def test_region_center_uses_border(self, bcgs):
        """With a border_bcg set, region_center uses it instead of primary."""
        sub = Subcluster(bcg_id=1, primary_bcg=bcgs[1], border_bcg=bcgs[5])
        assert abs(sub.region_center.ra.deg - 201.82) < 1e-10

    def test_display_center_always_primary(self, bcgs):
        """display_center always uses primary_bcg, even with border set."""
        sub = Subcluster(bcg_id=1, primary_bcg=bcgs[1], border_bcg=bcgs[5])
        assert abs(sub.display_center.ra.deg - 201.85) < 1e-10

    def test_bcg_1_5_grouping(self, bcgs):
        """The BCG 1+5 grouping scenario from the design discussion."""
        sub = Subcluster(
            bcg_id=1,
            primary_bcg=bcgs[1],       # identity comes from BCG 1
            border_bcg=bcgs[5],         # bisector boundary uses BCG 5
            member_bcgs=[bcgs[1], bcgs[5]],
            label="Main",
            color="white",
            group_id="A",
            group_members=(1, 5),
            is_dominant=True,
        )
        # Identity from primary
        assert sub.display_center.ra.deg == pytest.approx(201.85)
        assert sub.display_label == "Main"
        assert sub.display_color == "white"

        # Bisector boundary from border
        assert sub.region_center.ra.deg == pytest.approx(201.82)

        # Group info
        assert sub.group_members == (1, 5)
        assert len(sub.member_bcgs) == 2

    def test_group_label_overrides(self, bcgs):
        sub = Subcluster(
            bcg_id=1, primary_bcg=bcgs[1],
            label="1", color="white",
            group_label="Combined", group_color="gold",
        )
        assert sub.display_label == "Combined"
        assert sub.display_color == "gold"

    def test_from_config(self, bcgs):
        cfg = {
            "bcg_id": 1,
            "label": "Main",
            "color": "white",
            "radius_mpc": 2.0,
            "z_range": [0.42, 0.46],
            "border_bcg_id": 5,
            "member_bcg_ids": [1, 5],
            "group_id": "A",
            "group_members": [1, 5],
        }
        sub = Subcluster.from_config(cfg, bcgs)
        assert sub.bcg_id == 1
        assert sub.border_bcg.bcg_id == 5
        assert len(sub.member_bcgs) == 2
        assert sub.z_range == (0.42, 0.46)
        assert sub.group_members == (1, 5)

    def test_to_config_round_trip(self, bcgs):
        sub = Subcluster(
            bcg_id=1, primary_bcg=bcgs[1], border_bcg=bcgs[5],
            member_bcgs=[bcgs[1], bcgs[5]],
            label="Main", color="white", radius_mpc=2.5,
            z_range=(0.42, 0.46),
            group_id="A", group_members=(1, 5),
        )
        cfg = sub.to_config()
        assert cfg["bcg_id"] == 1
        assert cfg["border_bcg_id"] == 5
        assert cfg["member_bcg_ids"] == [1, 5]
        assert cfg["z_range"] == [0.42, 0.46]

    def test_spec_phot_members(self, bcgs):
        sub = Subcluster(bcg_id=1, primary_bcg=bcgs[1])
        assert sub.n_spec == 0
        assert sub.n_phot == 0

        sub.spec_members = pd.DataFrame({"RA": [201.0, 201.1], "Dec": [53.0, 53.1]})
        sub.phot_members = pd.DataFrame({"RA": [201.0], "Dec": [53.0]})
        assert sub.n_spec == 2
        assert sub.n_phot == 1
