#!/usr/bin/env python3
"""
Multi-species coordination + ion-pairing speciation + state frequencies + RDFs (MDAnalysis)

What you configure
------------------
1) Reference points ("the central species"):
   - ref_sel_str + ref_mode
   - Example: ref = cation atom positions, or ref = anion residue COMs

2) Species counters (list):
   Each species counter defines how to count "entities" around each reference.
   You can define:
     - atoms (ATOMWISE: counts atoms)
     - residues counted via atoms (RESIDUE_ANY / RESIDUE_ATLEAST(k) / RESIDUE_ALL)
     - residue_com (counts residues using COM; if selection is a subgroup, this becomes subgroup COM)

3) Pairing partners:
   Free/SSIP/CIP/AGG is computed by summing counts for the species you mark as pairing partners.

4) State frequencies:
   Builds a joint distribution of shell compositions:
     state = for each included species, (n_contact, n_shell2)
   where n_shell2 = n_outer - n_contact, outer defined by that species' cut2.

Outputs
-------
- timeseries.csv
- summary.txt
- state_frequencies.csv
- rdf__*.csv (one per enabled counter, matching how that counter defines distances)
- rdf__total__ref_vs_allOtherAtoms.csv (optional)

Notes
-----
- Distances are in Angstrom.
- PBC is handled.
- Optional unwrap transformation improves residue COM reliability if molecules are split across PBC.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from tqdm import tqdm

from .config import AnalysisConfig

# =============================================================================
# --- USER INPUTS --------------------------------------------------------------
# =============================================================================

# Input files
topology_file = "traj.tpr"
trajectory_file = "traj.xtc"

#print(f"Found {topology_file} and {trajectory_file} files to analyze")

# Output folder
output_dir = Path("speciation_outputs")

#print(f"The directory where the output files will be saves is {output_dir}")

# Trajectory range (optional)
start_frame: Optional[int] = None
stop_frame: Optional[int] = None
step_frame: int = 1

#DOUBT: in which units?

#print(f"This analysis set will be performed from {start_frame} to {stop_frame} , every {step_frame} frames.")


# Unwrapping (helps COM reliability when molecules are split across the box)
apply_unwrap: bool = True
guess_bonds_if_missing: bool = False  # use with care

#print(f"Trajectory unwraping is {apply_unwrap} and guess-bonds-if-missing option is {guess_bonds_if_missing}")

# Reference definition (central species)
ref_sel_str = "resname Li"  # EDIT ME
ref_mode: Literal["atoms", "residue_com"] = "atoms"


# Ion-pairing classification cutoffs (shared across all pairing partner species)
pair_cut1 = 3.0  # contact
pair_cut2 = 8.0  # outer boundary (needed for SSIP/Free)


# State-frequency controls
include_shell2_in_states = False  # you asked to include shell2
count_cap: Optional[int] = 4     # cap counts at this number (4 means "4+"); set None to disable
top_states_n = 15                # print and save top N most common states


# RDF controls
rdf_default_rmax = 20.0
rdf_default_nbins = 1000
compute_total_rdf = True  # ref vs all other atoms (atom-mode)


# Save counts instead of fractions for Free/SSIP/CIP/AGG?
write_counts_instead_of_fractions = False


# ---- Species counters --------------------------------------------------------
# Add as many as you want. You decide which are "pairing partners".
#
# Fields:
#   name: string label
#   target_sel: MDAnalysis selection
#   target_mode: "atoms" or "residue_com"
#   rule (only for target_mode="atoms"):
#       - "POINT"            => counts one point of three coordinaties (the com)
#       - "ATOMWISE"         => counts atoms
#       - "RESIDUE_ANY"      => counts residues with >=1 selected atom in cutoff
#       - "RESIDUE_ATLEAST"  => counts residues with >=k selected atoms in cutoff
#       - "RESIDUE_ALL"      => counts residues only if ALL selected atoms in that residue are in cutoff
#   atleast_k: integer for RESIDUE_ATLEAST
#   cut1, cut2: cutoffs in Angstrom (cut2 optional but required if you want shell2)
#   enabled: include in calculations
#   include_in_state: include in state-frequency definition
#   pairing_partner: include in Free/SSIP/CIP/AGG partner counts
#   compute_rdf: write an RDF that matches this counting definition
#
species_specs = [
    # Example: anion type (pairing partner) counted by donor atoms per residue (best for chemistry)
    dict(
        name="FSI",
        target_sel="resname FSI",       # EDIT ME
        target_mode="residue_com",
        rule="POINT",
        atleast_k=1,
        cut1=pair_cut1,
        cut2=pair_cut2,
        enabled=True,
        include_in_state=False,
        pairing_partner=False,
        compute_rdf=True,
    ),


    dict(
        name="FSI_solv",
        target_sel="resname FSI and name O*",       # EDIT ME
        target_mode="atoms",
        rule="RESIDUE_ANY",
        atleast_k=1,
        cut1=pair_cut1,
        cut2=pair_cut2,
        enabled=True,
        include_in_state=True,
        pairing_partner=True,
        compute_rdf=True,
    ),



    # Example: glyme solvent molecules counted as residues via donor atoms (RESIDUE_ANY as you requested)
    dict(
        name="DME_solv",
        target_sel="resname DME and name O*",         # EDIT ME
        target_mode="atoms",
        rule="RESIDUE_ANY",
        atleast_k=1,
        cut1=pair_cut1,
        cut2=pair_cut2,   # provide cut2 if you want shell2 counts for solvent in states
        enabled=True,
        include_in_state=True,
        pairing_partner=False,
        compute_rdf=True,
    ),

        dict(
        name="TOL",
        target_sel="resname TOL",       # EDIT ME
        target_mode="residue_com",
        rule="POINT",
        atleast_k=1,
        cut1=pair_cut1,
        cut2=pair_cut2,
        enabled=True,
        include_in_state=True,
        pairing_partner=False,
        compute_rdf=True,
    ),

    
    dict(
        name="FSI_oxygencount",
        target_sel="resname FSI and name O*",       # EDIT ME
        target_mode="atoms",
        rule="ATOMWISE",
        atleast_k=1,
        cut1=pair_cut1,
        cut2=pair_cut2,
        enabled=True,
        include_in_state=False,
        pairing_partner=False,
        compute_rdf=True,
    ),

    dict(
        name="DME_oxygencount",
        target_sel="resname DME and name O*",       # EDIT ME
        target_mode="atoms",
        rule="ATOMWISE",
        atleast_k=1,
        cut1=pair_cut1,
        cut2=pair_cut2,
        enabled=True,
        include_in_state=False,
        pairing_partner=False,
        compute_rdf=True,
    ),

    dict(
        name="FSI_nitrogencount",
        target_sel="resname FSI and name N*",       # EDIT ME
        target_mode="atoms",
        rule="ATOMWISE",
        atleast_k=1,
        cut1=pair_cut1,
        cut2=pair_cut2,
        enabled=True,
        include_in_state=False,
        pairing_partner=False,
        compute_rdf=True,
    ),    
    
]


# =============================================================================


PointMode = Literal["atoms", "residue_com"]
Rule = Literal["POINT", "ATOMWISE", "RESIDUE_ANY", "RESIDUE_ATLEAST", "RESIDUE_ALL"]


@dataclass(frozen=True)
class Cutoffs:
    r1: float
    r2: Optional[float] = None


@dataclass
class SpeciesCounter:
    name: str
    target_sel: str
    target_mode: PointMode
    rule: Rule
    atleast_k: int
    cutoffs: Cutoffs
    enabled: bool
    include_in_state: bool
    pairing_partner: bool
    compute_rdf: bool

    # Cached objects (filled after Universe exists)
    target_ag: Optional[mda.AtomGroup] = None
    required_per_res: Optional[np.ndarray] = None  # used for RESIDUE_ALL

    def attach(self, u: mda.Universe) -> None:
        """Cache AtomGroup and any per-residue requirements."""
        self.target_ag = u.select_atoms(self.target_sel)
        if self.target_ag.n_atoms == 0:
            warnings.warn(f"[{self.name}] selection returned 0 atoms: {self.target_sel}")

        if self.target_mode == "residue_com":
            self.rule = "POINT"  # force consistent semantics

#DOUBT: WHY? is it a double check?

        if self.target_mode == "atoms" and self.rule == "RESIDUE_ALL":
            req = np.zeros(u.residues.n_residues, dtype=int)
            # number of selected atoms per residue (Universe residue index)
            np.add.at(req, self.target_ag.resindices, 1)
            self.required_per_res = req

    def validate_for_pairing(self) -> None:
        """Enforce rules needed for Free/SSIP/CIP/AGG."""
        if not self.pairing_partner:
            return
        if self.cutoffs.r2 is None:
            raise ValueError(f"[{self.name}] pairing_partner requires cut2 (needed for SSIP/Free).")
        if self.target_mode == "atoms" and self.rule == "ATOMWISE":
            raise ValueError(
                f"[{self.name}] pairing_partner cannot use ATOMWISE; "
                "CIP/AGG are defined by number of partner entities (residues), not atoms."
            )


def build_universe(top: str, traj: str) -> mda.Universe:
    """
    Docstring for build_universe
    This function loads the trajecyory, unwraps it and guess bonds if missing.
    :param top: Description
    :type top: str
    :param traj: Description
    :type traj: str
    :return: Description
    :rtype: Universe
    """
    u = mda.Universe(top, traj)

    if guess_bonds_if_missing:
        try:
            u.atoms.guess_bonds()
        except Exception as exc:
            warnings.warn(f"Bond guessing failed ({exc}). Continuing without guessed bonds.")

    if apply_unwrap:
        try:
            from MDAnalysis.transformations import unwrap
            u.trajectory.add_transformations(unwrap(u.atoms))
        except Exception as exc:
            warnings.warn(
                "Unwrap transformation could not be applied. Residue COMs may be less reliable.\n"
                f"Reason: {exc}"
            )
    return u


def get_points_from_ag(ag: mda.AtomGroup, mode: PointMode) -> np.ndarray:
    """
    This function get the coordinates of either the atoms or the com of 
    the residues we are looking at according with the built dictionaries and stores
    them in a np array.
    :param ag: Description
    :type ag: mda.AtomGroup
    :param mode: Description
    :type mode: PointMode
    :return: Description
    :rtype: ndarray
    """
    
    if ag.n_atoms == 0:
        return np.empty((0, 3), dtype=float)
    if mode == "atoms":
        return ag.positions.copy()
    if mode == "residue_com":
        return ag.center_of_mass(compound="residues")
    raise ValueError(f"Unknown mode: {mode}")


def count_entities(
    u: mda.Universe,
    ref_pts: np.ndarray,
    counter: SpeciesCounter,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    This function counts the entities within cut1 and optionally within cut2 around each reference point,
    and save the counts in one tuple of either 1 or 2 np arrays.

    Returns:
      counts1: (Nref,) entities within r1
      counts2: (Nref,) entities within r2 (includes those in r1), or None if r2 not defined
    """
    
    #This line and the conditionals counts the number of ref atoms (Li) and created the np array filled with zeros for the target atoms.
    #This is a "check", in case there's no n_ref atoms, target_ag atoms or the n_atoms of target_ag is zero
    n_ref = ref_pts.shape[0]
    #print(n_ref)
    #print(counter.target_ag)
    if n_ref == 0:
        return np.zeros(0, dtype=int), None

    if counter.target_ag is None:
        raise RuntimeError("Counter is not attached to Universe.")

    if counter.target_ag.n_atoms == 0:
        zeros = np.zeros(n_ref, dtype=int)
        return zeros, (zeros.copy() if counter.cutoffs.r2 is not None else None)
        
    #Assigning the box dimensions to the box variable and the cutoff values to the variables r1 and r2
    box = u.dimensions
    r1 = counter.cutoffs.r1
    r2 = counter.cutoffs.r2

    # Build target points, here we are creating a brand-new array of zeros and filling it with counts
    if counter.target_mode == "residue_com":
        tgt_pts = get_points_from_ag(counter.target_ag, "residue_com")
        #print(tgt_pts)
        if tgt_pts.shape[0] == 0:
            zeros = np.zeros(n_ref, dtype=int) #creates an array of n_ref zeros (75 in this case)
            return zeros, (zeros.copy() if r2 is not None else None)

        if r2 is None:
            pairs = capped_distance(ref_pts, tgt_pts, max_cutoff=r1, box=box, return_distances=False)
            #print(pairs)
            i_ref = pairs[:,0] #counts of any Li (ref_pts) with any atom/com (tgt_pts). Eg: [0, 0, 5, 22, 22, 22] tells us Lithium #0 found 2 things, Lithium #5 found 1 thing, and Lithium #22 found 3 things
            #print(i_ref)
            counts1 = np.bincount(i_ref, minlength=n_ref).astype(int) #counts from i_ref in a way that agrupates the counts by the ref_pts index. Eg: index 0 -> 2, index 5 -> 1, index 22 -> 3
            #print(counts1)
            # minlength=n_ref ensures that even if i_ref from #1 through #4 found nothing, you still have numbers in the list #1, #2, #3, and #4 sitting there.
            # astype(int) converts the list to integers, as you can not have 2.5 molecules
            return counts1, None

        # Now instead of only pairs we have also dist (because we have set return_distances to True), so we are not only getting pairing counts, but the pairing distances
        # pairs: The 2D table of indices (Who found Whom).
        # dist: A 1D list of the actual distances.
        pairs, dist = capped_distance(ref_pts, tgt_pts, max_cutoff=r2, box=box, return_distances=True)
        i_ref = pairs[:,0]
        counts2 = np.bincount(i_ref, minlength=n_ref).astype(int)
        # numpy broadcasting, it's like a filter: list of boleans (True or False, according to dist < r1 or not)
        mask1 = dist < r1
        counts1 = np.bincount(i_ref[mask1], minlength=n_ref).astype(int)
        #print(f"Counts 1 {counts1}")
        #print(f"Counts 2 {counts2}")
        return counts1, counts2

    # Atom-mode targets (it's like the else of the if block counter.target_mode == "residue_com"):
    tgt_pts = counter.target_ag.positions
    #print(tgt_pts)
    if tgt_pts.shape[0] == 0:
        zeros = np.zeros(n_ref, dtype=int)
        return zeros, (zeros.copy() if r2 is not None else None)

    def residue_aggregate_counts(i_ref: np.ndarray, j_atom: np.ndarray, rule: Rule, atleast_k: int) -> np.ndarray:
        """
        This function groups atoms into whole molecules (residues)
        i_ref are the ref (Li) atom list
        j_atom are the contact (O, N, etc.) atom (from any molecule) list
        Vectorized residue-aggregation:
          - build unique pairs (ref_index, residue_index) and counts per pair
          - apply rule by masking those pairs
          - aggregate by ref_index via bincount
        """
        if i_ref.size == 0:
            return np.zeros(n_ref, dtype=int)

        res_idx = counter.target_ag.resindices[j_atom] # residences it's a numpy array to keep record of which residue every atom (in contact with the ref) belongs to
        #print(res_idx)
        pair_keys = np.column_stack((i_ref, res_idx)) # np.column_stack: Stitches the Lithium IDs and Molecule IDs side-by-side of the contact atom. Each atom index with each atom residue in contact
        #print(pair_keys)
        # uniq_pairs and pair_counts are the outputs that the np.unique function gives 
        uniq_pairs, pair_counts = np.unique(pair_keys, axis=0, return_counts=True) # removes duplicate and counts them
        #print(uniq_pairs) # simplified, so for example if 1 n_ref (Li) is in contact with two O aotms from the same molecule, it's in contact with that molecule twice, but we want to simplify and say "this Li is in contact with this molecule"
        #print(pair_counts) # here, we provide the counts for each simplified contact (uniw_pairs), for example if we had two O atoms from one molecule, the count for that contact will be 2
        
        
        if rule == "RESIDUE_ANY":
            mask = np.ones_like(pair_counts, dtype=bool) # np.ones_like is a NumPy shortcut to create a new array that is the exact same size as pair_counts, but fills every slot with True (represented by the number 1)
        elif rule == "RESIDUE_ATLEAST":
            mask = pair_counts >= atleast_k
        elif rule == "RESIDUE_ALL":
            if counter.required_per_res is None:
                raise ValueError("RESIDUE_ALL requires required_per_res.")
            required = counter.required_per_res[uniq_pairs[:, 1]]
            mask = pair_counts >= required
        else:
            raise ValueError(f"Invalid residue rule: {rule}")

        return np.bincount(uniq_pairs[mask, 0], minlength=n_ref).astype(int)


    if r2 is None:
        pairs = capped_distance(ref_pts, tgt_pts, max_cutoff=r1, box=box, return_distances=False)
        i_ref, j_atom = pairs[:,0], pairs[:,1] # ref indices in the first column and target atom indices in the second column

        if counter.rule == "ATOMWISE":
            counts1 = np.bincount(i_ref, minlength=n_ref).astype(int)
        else:
            counts1 = residue_aggregate_counts(i_ref, j_atom, counter.rule, counter.atleast_k)

        return counts1, None

    pairs, dist = capped_distance(ref_pts, tgt_pts, max_cutoff=r2, box=box, return_distances=True)
    i_ref_all, j_atom_all = pairs[:,0], pairs[:,1]

    # Outer counts
    if counter.rule == "ATOMWISE":
        counts2 = np.bincount(i_ref_all, minlength=n_ref).astype(int)
    else:
        counts2 = residue_aggregate_counts(i_ref_all, j_atom_all, counter.rule, counter.atleast_k)

    # Contact counts (subset)
    mask1 = dist < r1 # creates a list of True/False
    i_ref_1 = i_ref_all[mask1] # n_ref indices that found atoms closer than r1
    #print(i_ref_1)
    j_atom_1 = j_atom_all[mask1] # atom indices (from target_ag) that found atoms closer than r1
    #print(j_atom_1)
    if counter.rule == "ATOMWISE":
        counts1 = np.bincount(i_ref_1, minlength=n_ref).astype(int)
    else:
        counts1 = residue_aggregate_counts(i_ref_1, j_atom_1, counter.rule, counter.atleast_k)

    return counts1, counts2


def classify_ion_pairing(n_contact: np.ndarray, n_outer: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Free / SSIP / CIP / AGG based on partner totals.
      CIP: n_contact == 1
      AGG: n_contact >= 2
      SSIP: n_contact == 0 and (n_outer - n_contact) >= 1
      Free: n_outer == 0
    Refered to the cation
    n_contact counts the anions
    """
    n_shell = n_outer - n_contact
    is_cip = n_contact == 1
    is_agg = n_contact >= 2
    is_ssip = (n_contact == 0) & (n_shell >= 1)
    is_free = n_outer == 0
    return is_free, is_ssip, is_cip, is_agg


@dataclass
class RDFAccumulator:
    """
    RDF accumulator using capped_distance (no full distance matrix).
    Normalization:
      expected(bin) per frame = N_ref * (N_tgt / V) * shell_volume(bin)
      g(r) = observed / expected
    """
    label: str
    ref_ag: mda.AtomGroup
    ref_mode: PointMode
    tgt_ag: mda.AtomGroup
    tgt_mode: PointMode
    rmax: float
    nbins: int
    exclude_ref_from_tgt: bool = False

    def __post_init__(self):
        """
        This function inside the dataclass specific RDF calculator creates 
        a counts array and an expected array and fill them with zeros so we are ready
        to start counting when the simulation begins.
        """
        self.edges = np.linspace(0.0, self.rmax, self.nbins + 1) # get a list of evenly spaced numbers every 0.1 A
        #print(self.edges)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:]) # get the center point between every pair of marks
        #print(self.centers)
        self.shell_volumes = (4.0 / 3.0) * np.pi * (self.edges[1:] ** 3 - self.edges[:-1] ** 3) # calculates the volume of every single shell
        #print(self.shell_volumes)

        self.counts = np.zeros(self.nbins, dtype=np.float64) # very precise decimals
        #print(self.counts) # this is the counts for the observed
        self.expected = np.zeros(self.nbins, dtype=np.float64) # very precise decimals
        #print(self.counts) # this is the counts for the expected

    # self is the specific instance of the class
    # it's the way the class refers to this specific example, i.e.: Li-FSI


    def accumulate(self, u: mda.Universe) -> None:
        """
        This function inside the dataclass specific RDF calculator adds points
        to the histogram
        """
        ref_pts = get_points_from_ag(self.ref_ag, self.ref_mode)
        if ref_pts.shape[0] == 0: #if the numbers of rows is zero, shape is the dimension of the table (rows, columns) and 0 in brackets (slicing) refers to the rows
            return

        if self.exclude_ref_from_tgt:
            if self.ref_mode != "atoms" or self.tgt_mode != "atoms":
                warnings.warn(f"[{self.label}] exclude_ref_from_tgt intended for atom-mode RDFs.")
            tgt_ag = self.tgt_ag.difference(self.ref_ag)
            #print(tgt_ag)
            tgt_pts = tgt_ag.positions.copy()
        else:
            tgt_pts = get_points_from_ag(self.tgt_ag, self.tgt_mode)

        if tgt_pts.shape[0] == 0:
            return

        box = u.dimensions
        lx, ly, lz = box[0], box[1], box[2]
        V = lx * ly * lz

        # observed data collection
        pairs, dist = capped_distance(ref_pts, tgt_pts, max_cutoff=self.rmax, box=box, return_distances=True)
        if dist.size > 0:
            hist, _ = np.histogram(dist, bins=self.edges)
            self.counts += hist

        n_ref = ref_pts.shape[0]
        n_tgt = tgt_pts.shape[0]
        self.expected += (n_ref * (n_tgt / V)) * self.shell_volumes # this is what we would expect if the system is perfectly random
        # this becomes a normalization operation, we will use this ideal random distribution as reference
        # it's total density for all the Li or ref atoms, that's why we multiply by n_ref
        #print(self.expected)

    def to_dataframe(self) -> pd.DataFrame:
        """
        This method is the "Final Report" stage. It takes the two tally sheets 
        we ve been filling for hundreds of frames (self.counts and self.expected) 
        and performs the final division to create the g(r) curve.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(self.expected > 0, self.counts / self.expected, np.nan)
        return pd.DataFrame({"r_A": self.centers, "g_r": g}) # function from pandas library to creare a table using the described dictionary


def cap_array(x: np.ndarray, cap: Optional[int]) -> np.ndarray:
    """
    This function is used for the "State Frequencies" analysis. 
    If a Lithium ion is surrounded by 10 water molecules, 
    sometimes we don't care if it's 10, 11, or 12, we just want to say it's "4 or more."
    """
    if cap is None:
        return x
    return np.minimum(x, cap).astype(int)


def state_string(spec_order: List[SpeciesCounter], row: np.ndarray, cap: Optional[int]) -> str:
    """
    Build a readable state string from a row containing [s1_c, s1_sh, s2_c, s2_sh, ...].
    c: get the contact count from the current position
    sh: get the Shell 2 count from the next position.
    idx += 2: jump forward two steps in the list so we are ready for the next molecule.
    The data is taken later within the main function. This function it's just a toll
    that will be used afterwards in the main() function.
    """
    parts = [] # we define a list that will store the labels of the molecules
    idx = 0
    for sp in spec_order:
        c = int(row[idx]); sh = int(row[idx + 1]); idx += 2
        def fmt(v: int) -> str:
            if cap is None:
                return str(v)
            return f"{cap}+" if v >= cap else str(v)
        parts.append(f"{sp.name}[c={fmt(c)},sh={fmt(sh)}]")
    return ";".join(parts)


def run_analysis(config: AnalysisConfig) -> None:
    """Run an analysis described by ``config`` and write its result files.

    Distances in the configuration are interpreted as Ångström, matching
    MDAnalysis coordinate units.
    """
    global topology_file, trajectory_file, output_dir
    global start_frame, stop_frame, step_frame
    global apply_unwrap, guess_bonds_if_missing, ref_sel_str, ref_mode
    global pair_cut1, pair_cut2, include_shell2_in_states, count_cap, top_states_n
    global rdf_default_rmax, rdf_default_nbins, compute_total_rdf
    global write_counts_instead_of_fractions, species_specs

    topology_file = str(config.topology_file)
    trajectory_file = str(config.trajectory_file)
    output_dir = config.output_dir
    start_frame, stop_frame, step_frame = config.start_frame, config.stop_frame, config.step_frame
    apply_unwrap = config.apply_unwrap
    guess_bonds_if_missing = config.guess_bonds_if_missing
    ref_sel_str, ref_mode = config.ref_sel_str, config.ref_mode
    pair_cut1, pair_cut2 = config.pair_cut1, config.pair_cut2
    include_shell2_in_states = config.include_shell2_in_states
    count_cap, top_states_n = config.count_cap, config.top_states_n
    rdf_default_rmax, rdf_default_nbins = config.rdf_default_rmax, config.rdf_default_nbins
    compute_total_rdf = config.compute_total_rdf
    write_counts_instead_of_fractions = config.write_counts_instead_of_fractions
    species_specs = [species.as_dict() for species in config.species]

    output_dir.mkdir(parents=True, exist_ok=True)

    u = build_universe(topology_file, trajectory_file)

    # Reference AtomGroup cached once (coordinates update every frame)
    ref_ag = u.select_atoms(ref_sel_str)
    if ref_ag.n_atoms == 0:
        raise RuntimeError(f"Reference selection returned 0 atoms: {ref_sel_str}")

    # Build counters and attach to Universe
    counters: List[SpeciesCounter] = []
    for d in species_specs:
        if not d.get("enabled", True):
            continue
        c = SpeciesCounter(
            name=d["name"],
            target_sel=d["target_sel"],
            target_mode=d["target_mode"],
            rule=d.get("rule", "POINT"),
            atleast_k=int(d.get("atleast_k", 1)),
            cutoffs=Cutoffs(float(d["cut1"]), (float(d["cut2"]) if d.get("cut2") is not None else None)),
            enabled=True,
            include_in_state=bool(d.get("include_in_state", True)),
            pairing_partner=bool(d.get("pairing_partner", False)),
            compute_rdf=bool(d.get("compute_rdf", True)),
        )
        c.attach(u)
        counters.append(c)
        #print(c)

    if not counters:
        raise RuntimeError("No enabled species counters. Check species_specs.")

    # Pairing partners (for Free/SSIP/CIP/AGG)
    pairing = [c for c in counters if c.pairing_partner] # list comprehension
    # Create a new list called pairing, fill it, look at every counter c in the counters list. If c.pairing_partner is True, add it to this new list
    if not pairing:
        raise RuntimeError("No pairing partners defined. Mark at least one species as pairing_partner=True.")

    # Enforce pairing requirements
    for c in pairing:
        c.validate_for_pairing()
        # Also enforce shared pairing cutoffs (you asked to keep Free/SSIP/CIP/AGG meaningful)
        if abs(c.cutoffs.r1 - pair_cut1) > 1e-6 or abs((c.cutoffs.r2 or 0) - pair_cut2) > 1e-6:
            warnings.warn(
                f"[{c.name}] pairing partner cutoffs differ from pair_cut1/pair_cut2. "
                "For consistent speciation across partner types, keep them identical."
            )
    
    #print(pairing)

    # State specs order, another list comprehension, it identifies which molecules 
    # will be used to build the "State Strings" 
    state_specs = [c for c in counters if c.include_in_state]

    # If shell2 is requested, warn about any state specs missing cut2
    if include_shell2_in_states:
        for c in state_specs:
            if c.cutoffs.r2 is None:
                warnings.warn(f"[{c.name}] included in state, but cut2 is None -> shell2 will always be 0.")

    # RDF accumulators
    rdf_accs: List[RDFAccumulator] = []
    for c in counters:
        if not c.compute_rdf:
            continue
        # rmax should cover at least this counter's outer cutoff, or default
        needed_rmax = max(rdf_default_rmax, c.cutoffs.r1, (c.cutoffs.r2 or 0.0))
        rdf_accs.append(
            RDFAccumulator(
                label=f"rdf__ref__{ref_sel_str.replace(' ', '_')}__{c.name}",
                ref_ag=ref_ag,
                ref_mode=ref_mode,
                tgt_ag=c.target_ag,
                tgt_mode=c.target_mode,
                rmax=needed_rmax,
                nbins=rdf_default_nbins,
                exclude_ref_from_tgt=False,
            )
        )

    if compute_total_rdf:
        if ref_mode != "atoms":
            warnings.warn("compute_total_rdf=True but ref_mode!='atoms'. Skipping total RDF.")
        else:
            rdf_accs.append(
                RDFAccumulator(
                    label="rdf__total__ref_vs_allOtherAtoms",
                    ref_ag=ref_ag,
                    ref_mode="atoms",
                    tgt_ag=u.atoms,
                    tgt_mode="atoms",
                    rmax=rdf_default_rmax,
                    nbins=rdf_default_nbins,
                    exclude_ref_from_tgt=True,
                )
            )

    # Time series records
    records: List[Dict] = [] # store one dictionary for every frame (the "Time Series").

    # Global state counter: key = tuple of ints, value = count
    state_counter: Dict[Tuple[int, ...], int] = {} 

    # Trajectory loop
    frame_slice = slice(start_frame, stop_frame, step_frame)
    traj = u.trajectory[frame_slice]
    for ts in tqdm(traj, total=len(traj), desc="Analyzing trajectory", unit="frame"):
    #for ts in u.trajectory[frame_slice]:
        time_ps = float(ts.time)

        # Reference points for this frame
        ref_pts = get_points_from_ag(ref_ag, ref_mode)
        n_ref = ref_pts.shape[0]
        if n_ref == 0:
            continue

        # Per-counter counts stored here
        per_counter_counts: Dict[str, Dict[str, np.ndarray]] = {}

        # Compute counts for each species
        for c in counters:
            n1, n2 = count_entities(u, ref_pts, c) # n1 and n2 are count1 and count2 from function count_entities
            shell = (n2 - n1) if (n2 is not None) else np.zeros_like(n1)
            per_counter_counts[c.name] = {
                "contact": n1,
                "outer": (n2 if n2 is not None else np.zeros_like(n1)),
                "shell": shell,
                "has_outer": np.array([1 if c.cutoffs.r2 is not None else 0], dtype=int),
            }

        # ----- Ion-pairing speciation totals (sum over pairing partners) -------
        partner_contact_total = np.zeros(n_ref, dtype=int)
        partner_outer_total = np.zeros(n_ref, dtype=int)

        for c in pairing:
            partner_contact_total += per_counter_counts[c.name]["contact"]
            partner_outer_total += per_counter_counts[c.name]["outer"]

        is_free, is_ssip, is_cip, is_agg = classify_ion_pairing(partner_contact_total, partner_outer_total)

        counts = {
            "Free": int(is_free.sum()),
            "SSIP": int(is_ssip.sum()),
            "CIP": int(is_cip.sum()),
            "AGG": int(is_agg.sum()),
        }
        if write_counts_instead_of_fractions:
            frac_or_count = {k: float(v) for k, v in counts.items()}
        else:
            frac_or_count = {k: float(v) / float(n_ref) for k, v in counts.items()}

        # Build time series row
        row = {
            "frame": int(ts.frame),
            "time_ps": time_ps,
            "N_reference": int(n_ref),
            **frac_or_count,
            "CN_pairPartners_contact": float(partner_contact_total.mean()),
            "CN_pairPartners_outer": float(partner_outer_total.mean()),
            "CN_pairPartners_shell2": float((partner_outer_total - partner_contact_total).mean()),
        }

        # Mean CNs for every species counter
        for c in counters:
            row[f"{c.name}__mean_contact"] = float(per_counter_counts[c.name]["contact"].mean())
            if c.cutoffs.r2 is not None:
                row[f"{c.name}__mean_outer"] = float(per_counter_counts[c.name]["outer"].mean())
                row[f"{c.name}__mean_shell2"] = float(per_counter_counts[c.name]["shell"].mean())

        records.append(row)

        # ----- State frequencies (composition) --------------------------------
        if state_specs:
            cols = []
            for c in state_specs:
                contact = cap_array(per_counter_counts[c.name]["contact"], count_cap)
                shell = cap_array(per_counter_counts[c.name]["shell"], count_cap) if include_shell2_in_states else np.zeros_like(contact)
                cols.append(contact)
                cols.append(shell)
            state_matrix = np.column_stack(cols).astype(int)

            #print(state_specs)
            #print(shell)
            
            uniq, cnt = np.unique(state_matrix, axis=0, return_counts=True)
            for st_row, ccount in zip(uniq, cnt):
                key = tuple(int(x) for x in st_row)
                state_counter[key] = state_counter.get(key, 0) + int(ccount)

        # ----- RDF accumulation ----------------------------------------------
        for acc in rdf_accs:
            acc.accumulate(u)

    # ----------------------- Write timeseries -------------------------------
    df = pd.DataFrame.from_records(records)
    df.to_csv(output_dir / "timeseries.csv", index=False)

    # ----------------------- State frequencies ------------------------------
    if state_specs and state_counter:
        total_events = sum(state_counter.values())
        rows = []
        # Sort by count descending
        sorted_items = sorted(state_counter.items(), key=lambda kv: kv[1], reverse=True)

        for key, ccount in sorted_items:
            st_arr = np.array(key, dtype=int)
            st_str = state_string(state_specs, st_arr, count_cap)
            rows.append(
                {
                    "state": st_str,
                    "count": ccount,
                    "fraction": ccount / total_events,
                }
            )
        state_df = pd.DataFrame(rows)
        state_df.to_csv(output_dir / "state_frequencies.csv", index=False)

        # Top-N summary lines
        top_lines = []
        top_lines.append(f"Top {min(top_states_n, len(rows))} most common states (contact + shell2):")
        for i, r in enumerate(rows[:top_states_n], start=1):
            top_lines.append(f"{i:>2}. {100.0 * r['fraction']:.2f}%  |  {r['state']}")
        (output_dir / "top_states.txt").write_text("\n".join(top_lines))
    else:
        (output_dir / "top_states.txt").write_text("No state specs or no states counted.\n")

    # ----------------------- Summary ----------------------------------------
    summary = []
    summary.append(f"Frames analyzed: {len(df)}")
    summary.append(f"Reference: {ref_sel_str} (mode={ref_mode})")
    summary.append("")
    summary.append(f"Pairing cutoffs: cut1={pair_cut1} A, cut2={pair_cut2} A")
    summary.append("Pairing partner species: " + ", ".join([c.name for c in pairing]))
    summary.append("")

    for col in ["Free", "SSIP", "CIP", "AGG"]:
        if col in df.columns:
            val = df[col].mean()
            if write_counts_instead_of_fractions:
                summary.append(f"Mean {col} count per frame: {val:.3f}")
            else:
                summary.append(f"Mean {col} fraction: {100.0 * val:.2f}%")

    summary.append("")
    summary.append("Species counters:")
    for c in counters:
        summary.append(
            f"- {c.name}: mode={c.target_mode}, rule={c.rule}, k={c.atleast_k}, "
            f"cut1={c.cutoffs.r1}, cut2={c.cutoffs.r2}, "
            f"pairing_partner={c.pairing_partner}, include_in_state={c.include_in_state}"
        )

    # Append top states preview into summary.txt as well
    top_states_path = output_dir / "top_states.txt"
    if top_states_path.exists():
        summary.append("")
        summary.append(top_states_path.read_text().strip())

    (output_dir / "summary.txt").write_text("\n".join(summary))

    # ----------------------- RDF outputs ------------------------------------
    for acc in rdf_accs:
        rdf_df = acc.to_dataframe()
        safe = acc.label.replace(" ", "_").replace("/", "_")
        rdf_df.to_csv(output_dir / f"{safe}.csv", index=False)

    print(f"Done. Outputs in: {output_dir.resolve()}")
    print(f"- {output_dir / 'timeseries.csv'}")
    print(f"- {output_dir / 'summary.txt'}")
    print(f"- {output_dir / 'state_frequencies.csv'}")
    print(f"- {output_dir / 'top_states.txt'}")
    print(f"- RDF files: {len(rdf_accs)}")

    # ----------------------- Coordination number summary table ---------------
    # This summarizes trajectory-averaged coordination numbers per species.
    # It reports:
    #   CN_contact : mean number of entities in [0, cut1)
    #   CN_shell2  : mean number of entities in [cut1, cut2)  (only if cut2 exists)
    #   CN_outer   : mean number of entities in [0, cut2)     (only if cut2 exists)

    cn_rows = []

    for c in counters:
        # Columns exist only if cut2 was defined for that species
        col_contact = f"{c.name}__mean_contact"
        col_outer   = f"{c.name}__mean_outer"
        col_shell2  = f"{c.name}__mean_shell2"

        if col_contact not in df.columns:
            continue

        cn_contact = float(df[col_contact].mean())

        if c.cutoffs.r2 is not None and col_outer in df.columns and col_shell2 in df.columns:
            cn_outer  = float(df[col_outer].mean())
            cn_shell2 = float(df[col_shell2].mean())

            cn_rows.append({
                "species": c.name,
                "counting_mode": f"{c.target_mode}:{c.rule}" if c.target_mode == "atoms" else "residue_com:POINT",
                "cut_contact_A": f"[0, {c.cutoffs.r1:g})",
                "cut_shell2_A": f"[{c.cutoffs.r1:g}, {c.cutoffs.r2:g})",
                "cut_outer_A": f"[0, {c.cutoffs.r2:g})",
                "CN_contact": cn_contact,
                "CN_shell2": cn_shell2,
                "CN_outer": cn_outer,
            })
        else:
            cn_rows.append({
                "species": c.name,
                "counting_mode": f"{c.target_mode}:{c.rule}" if c.target_mode == "atoms" else "residue_com:POINT",
                "cut_contact_A": f"[0, {c.cutoffs.r1:g})",
                "cut_shell2_A": "",
                "cut_outer_A": "",
                "CN_contact": cn_contact,
                "CN_shell2": np.nan,
                "CN_outer": np.nan,
            })

    # Also include the aggregated pairing-partner CNs (if present)
    if "CN_pairPartners_contact" in df.columns:
        cn_rows.append({
            "species": "PAIR_PARTNERS_TOTAL",
            "counting_mode": "sum(pairing_partner species)",
            "cut_contact_A": f"[0, {pair_cut1:g})",
            "cut_shell2_A": f"[{pair_cut1:g}, {pair_cut2:g})",
            "cut_outer_A": f"[0, {pair_cut2:g})",
            "CN_contact": float(df["CN_pairPartners_contact"].mean()),
            "CN_shell2": float(df["CN_pairPartners_shell2"].mean()) if "CN_pairPartners_shell2" in df.columns else np.nan,
            "CN_outer": float(df["CN_pairPartners_outer"].mean()) if "CN_pairPartners_outer" in df.columns else np.nan,
        })

    cn_df = pd.DataFrame(cn_rows)

    # Order nicely
    preferred_cols = [
        "species", "counting_mode",
        "cut_contact_A", "cut_shell2_A", "cut_outer_A",
        "CN_contact", "CN_shell2", "CN_outer",
    ]
    cn_df = cn_df[preferred_cols]

    # Save
    cn_df.to_csv(output_dir / "coordination_numbers_summary.csv", index=False)

    # Print a readable view
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 140):
        print("\nCoordination number summary (trajectory averages):")
        print(cn_df.to_string(index=False))
        print(f"\nSaved: {output_dir / 'coordination_numbers_summary.csv'}")

    
