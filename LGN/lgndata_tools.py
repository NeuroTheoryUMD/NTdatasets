"""Portable read API for the built LGN Stage-1 dataset.

This is the ONE file meant to travel with the *data*, not the build
process -- if you copy `derived/` (the per-penetration `.pkl` files +
`stim_library.h5`) somewhere else, this single module is all you need
alongside it, nothing else from `lgn_tools/`. It doesn't care where its
own code lives; only where the data lives, via `set_data_root()`/`root=`
(see below). The scripts that *build* that data (`build_dataset_exptN.py`
and everything else in this folder, including `repeat_blocks.py`) are a
separate, one-repo-only concern and don't need to travel anywhere.

**Deliberately self-contained, not importing `repeat_blocks.py`** (2026-08,
by user design decision): the repeat-block-splitting logic in section 4
below (`_repeat_boundaries`/`_build_repeat_trial_blocked`) is a duplicate
of `repeat_blocks.py`'s functions of the same name (minus the leading
underscore), not a shared import. The read/interpret side of this project
is meant to be static and independent once the dataset is built -- it
shouldn't share live code with the build side, which may change if the
dataset is ever rebuilt. If the build-side splitting logic ever needs to
change, update both copies deliberately at that point; until then, the
duplication is intentional, not drift. This does NOT touch how repeats are
represented on disk (still the 3 shapes documented on `load_repeat_trial`)
-- only which code path reconstructs the `'flat'`-shape case at load time.

Three layers, merged into one file on purpose (previously
`session_index.py` + `cell_index.py` + `stim_lib.py` -- same functions,
just one import now instead of three):

1. **Numbered session access** (`build_session_table`, `name_for`,
   `number_for`). Raw experiment/penetration *names* remain the key inside
   every `.pkl` file and every raw recording -- but for someone just
   working through this dataset, a flat number is easier than remembering
   "c11" vs "d12" vs "e21". `experiment_number` spans all 5 experiments
   (0=Expt1..4=Expt5, stable regardless of what a given experiment does or
   doesn't have -- e.g. Expt1 has no NAT/SBN unique-pass data, but is still
   experiment_number 0). `penetration_number` is 0-indexed within its
   experiment, sorted-name order.

2. **Cell-first search** (`build_cell_trial_table`, `find_cells`,
   `cell_info`, `load_cell`). "What cells have long natural-scene data?" ->
   `find_cells(modality='NAT', regime='unique')` returns numbered
   `(experiment_number, penetration_number, cell_number)` handles, no
   names or trial-key strings required anywhere. "What data does *this*
   cell have?" -> `cell_info(experiment_number, penetration_number,
   cell_number)`. `load_cell(...)` loads just that one cell's data, given
   a `trial_key` that normally comes straight off a `find_cells()`/
   `cell_info()` row rather than being typed by hand.

3. **Trial-level access + the shared stimulus library**
   (`StimLibraryH5`, `load_trial`, `scan_all_trials`). The machinery
   everything above is built on -- useful directly if you want trial-level
   rather than cell-level control (e.g. loading every cell that shares one
   trial's stimulus at once).

4. **Repeat trials** (`load_repeat_trial`, `load_repeat_cell`,
   `compute_psth`). Repeat trials (`regime == 'repeat'`) are stored on disk
   in one of three different shapes depending on experiment/trial-type (see
   `load_repeat_trial`'s docstring) -- these functions normalize across all
   three so callers never need to know which one a given trial uses.
   `find_cells(regime='repeat', ...)` already covers *finding* repeats for a
   given cell/condition (nothing new needed there); these functions cover
   *loading* them into a uniform, PSTH-ready shape.

Data location: `set_data_root(path)` sets a session-wide default (every
function's `root=None` falls back to it); an explicit `root=`/`path=`
argument on any individual call always wins over that. If neither is ever
set, functions fall back to finding `derived/` two directories above this
file -- convenient zero-config behavior inside this repo, meaningless once
this file is copied elsewhere (there, `set_data_root()` is the answer).
"""

from __future__ import annotations

import os
import pickle

import h5py
import numpy as np
import pandas as pd

_FILE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_NAMES = ["Expt1", "Expt2", "Expt3", "Expt4", "Expt5"]

_default_root: str | None = None


def set_data_root(path: str) -> None:
    """Set the session-wide default data root (the directory containing
    `derived/`). Every function below falls back to this when its own
    `root=` argument is omitted."""
    global _default_root
    _default_root = os.path.abspath(path)


def get_data_root() -> str:
    """The current default: whatever `set_data_root()` set, else a
    `__file__`-relative guess (works only when this file hasn't been moved
    out of the repo it was built in)."""
    return _default_root if _default_root is not None else _FILE_ROOT


def _resolve_root(root: str | None) -> str:
    return root if root is not None else get_data_root()


def _dataset_dir(experiment: str, root: str | None = None) -> str:
    return os.path.join(_resolve_root(root), "derived", experiment, "dataset")


# ---------------------------------------------------------------------------
# 1. Numbered session access
# ---------------------------------------------------------------------------

_session_table_cache: dict[str, pd.DataFrame] = {}


def build_session_table(root: str | None = None) -> pd.DataFrame:
    """One row per penetration across all 5 experiments. Columns:
    session_index (flat, 0..N-1), experiment_number (0-4), experiment_name,
    penetration_number (0..k-1 within its experiment), penetration_name,
    session_date."""
    rows = []
    session_index = 0
    for experiment_number, experiment_name in enumerate(EXPERIMENT_NAMES):
        d = _dataset_dir(experiment_name, root)
        pen_names = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".pkl"))
        for penetration_number, pen in enumerate(pen_names):
            with open(os.path.join(d, f"{pen}.pkl"), "rb") as f:
                meta = pickle.load(f)["meta"]
            rows.append({
                "session_index": session_index,
                "experiment_number": experiment_number,
                "experiment_name": experiment_name,
                "penetration_number": penetration_number,
                "penetration_name": pen,
                "session_date": meta.get("session_date") or meta.get("date"),
            })
            session_index += 1
    return pd.DataFrame(rows)


def _cached_session_table(root: str | None = None) -> pd.DataFrame:
    key = _resolve_root(root)
    if key not in _session_table_cache:
        _session_table_cache[key] = build_session_table(root)
    return _session_table_cache[key]


def name_for(experiment_number: int, penetration_number: int, root: str | None = None) -> tuple[str, str]:
    """(experiment_number, penetration_number) -> (experiment_name, penetration_name)."""
    t = _cached_session_table(root)
    row = t[(t.experiment_number == experiment_number) & (t.penetration_number == penetration_number)]
    if len(row) == 0:
        raise KeyError(f"no penetration at experiment_number={experiment_number}, penetration_number={penetration_number}")
    r = row.iloc[0]
    return r.experiment_name, r.penetration_name


def number_for(experiment_name: str, penetration_name: str, root: str | None = None) -> tuple[int, int]:
    """(experiment_name, penetration_name) -> (experiment_number, penetration_number) -- the reverse lookup, for tracing a number back to the raw data."""
    t = _cached_session_table(root)
    row = t[(t.experiment_name == experiment_name) & (t.penetration_name == penetration_name)]
    if len(row) == 0:
        raise KeyError(f"no penetration named {experiment_name}/{penetration_name}")
    r = row.iloc[0]
    return int(r.experiment_number), int(r.penetration_number)


# ---------------------------------------------------------------------------
# 3. Trial-level access + the shared stimulus library (cell-first search
#    below is built on this, hence defined first)
# ---------------------------------------------------------------------------

class StimLibraryH5:
    """Opens stim_library.h5 once; resolves stim_ref dicts to arrays.

    Caches materialized slices in-process, keyed by (family, start, stop),
    so multiple trials/neurons referencing the identical range only pay the
    HDF5 read + array-materialization cost once.
    """

    def __init__(self, path: str | None = None, root: str | None = None):
        self.path = path or os.path.join(_resolve_root(root), "derived", "stim_library.h5")
        self._file = h5py.File(self.path, "r")
        self._cache: dict[tuple[str, int, int], np.ndarray] = {}

    def resolve(self, stim_ref: dict | None) -> np.ndarray | None:
        if stim_ref is None:
            return None
        key = (stim_ref["family"], stim_ref["start"], stim_ref["stop"])
        if key not in self._cache:
            family, start, stop = key
            self._cache[key] = self._file[family][start:stop]
        return self._cache[key]

    def clear_cache(self):
        self._cache.clear()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _trial_rows(expt: str, root: str | None = None):
    """Yield one summary dict per trial across every penetration in `expt`,
    covering both unique and repeat regimes. Cheap now that .pkl files no
    longer embed the large NAT/SBN arrays."""
    d = _dataset_dir(expt, root)
    for fname in sorted(os.listdir(d)):
        if not fname.endswith(".pkl"):
            continue
        pen = fname[:-4]
        with open(os.path.join(d, fname), "rb") as f:
            ds = pickle.load(f)
        n_units = len(ds["units"])
        for trial_key, t in ds["trials"].items():
            stim_ref = t.get("stim_ref")
            has_embedded_stim = t.get("stim") is not None
            yield {
                "experiment": expt,
                "penetration": pen,
                "trial_key": trial_key,
                "modality": t.get("modality"),
                "code": t.get("code"),
                "regime": t.get("regime"),
                "contrast": t.get("contrast"),
                "contrast_rms": t.get("contrast_rms"),
                "stim_ref": stim_ref,
                "has_stim": has_embedded_stim or stim_ref is not None,
                "n_units": n_units,
                "notes": " | ".join(t.get("notes", [])),
            }


def scan_all_trials(experiments: list[str] | None = None, root: str | None = None) -> pd.DataFrame:
    """One DataFrame across all experiments/regimes, `experiment`/
    `penetration` given as names (this is a human-browsable summary table,
    not the primary per-trial access path -- see `load_trial`/`find_cells`
    for that). `experiment_number`/`penetration_number` columns are also
    included so a specific row can be handed straight to `load_trial`.
    `stim_ref` is populated for NAT/SBN trials whose content lives in
    stim_library.h5; `has_stim` also covers FF trials, whose (small) array
    stays embedded in the .pkl and isn't routed through this loader."""
    rows = []
    for expt in experiments or EXPERIMENT_NAMES:
        rows.extend(_trial_rows(expt, root))
    df = pd.DataFrame(rows)
    if len(df):
        sessions = _cached_session_table(root)[["experiment_name", "penetration_name", "experiment_number", "penetration_number"]]
        df = df.merge(sessions, left_on=["experiment", "penetration"], right_on=["experiment_name", "penetration_name"], how="left")
        df = df.drop(columns=["experiment_name", "penetration_name"])
    return df


def load_trial(experiment_number: int, penetration_number: int, trial_key: str, lib: StimLibraryH5 | None = None, root: str | None = None) -> dict:
    """Load one trial's full data, addressed by numbered session
    (`number_for(name, name)` gives you these numbers if you're starting
    from raw experiment/penetration names instead). Resolves stim_ref via
    `lib` if present, else falls back to any embedded `stim` (FF trials).
    Pass a shared `lib` across multiple calls to get the caching benefit; a
    temporary one is opened/closed automatically if omitted."""
    experiment, penetration = name_for(experiment_number, penetration_number, root)
    path = os.path.join(_dataset_dir(experiment, root), f"{penetration}.pkl")
    with open(path, "rb") as f:
        ds = pickle.load(f)
    t = ds["trials"][trial_key]

    owns_lib = lib is None
    if owns_lib:
        lib = StimLibraryH5(root=root)
    try:
        stim = lib.resolve(t.get("stim_ref")) if t.get("stim_ref") is not None else t.get("stim")
    finally:
        if owns_lib:
            lib.close()

    return {
        "stim": stim,
        # stim_ref: {"family", "start", "stop"} -- exactly where `stim` sits within
        # the shared WN/NAT family sequence (None for FF trials, whose small stim
        # is embedded directly rather than referenced). Two trials with the same
        # `family` and `start` are already in register frame-for-frame -- no
        # shifting needed, just truncate to the shorter `stop` when comparing
        # trials of different lengths. A different `start` (e.g. Expt5's
        # wntest2/wntest3, which begin at frame 18000/36000 of the WN family
        # rather than 0) means a real offset -- subtract the difference in
        # `start` to align. See CLAUDE.md "Are all stimuli the same length" for
        # the full distribution across this dataset.
        "stim_ref": t.get("stim_ref"),
        "frame_times": t.get("frame_times"),
        "spk_times": t.get("spk_times"),
        "per_unit": t.get("per_unit"),
        "unit_names": [u["name"] for u in ds["units"]],
        # xy_class (index-aligned with unit_names/spk_times): the field to use --
        # privileges the phase-grating measurement over any lab spreadsheet label
        # where both exist, see lgn_tools/xy_classification.py (build-time only --
        # the result is what's stored here). xy_class_log_f2f1 is the underlying
        # log(F2/F1) power ratio where xy_class came from that measurement (None
        # otherwise) -- values near 0 are borderline calls.
        "xy_class": [u.get("xy_class") for u in ds["units"]],
        "xy_class_source": [u.get("xy_class_source") for u in ds["units"]],
        "xy_class_log_f2f1": [u.get("xy_class_log_f2f1") for u in ds["units"]],
        # contrast: 'HC'/'MC'/'LC'/None; contrast_rms: the RMS contrast value
        # where known (None for Expt4's still-undocumented MC) -- see
        # CLAUDE.md "Contrast metadata" for provenance per trial type.
        "contrast": t.get("contrast"),
        "contrast_rms": t.get("contrast_rms"),
        "contrast_notes": t.get("contrast_notes"),
        "regime": t.get("regime"),
        "modality": t.get("modality"),
        "notes": t.get("notes", []),
    }


# ---------------------------------------------------------------------------
# 2. Cell-first search
# ---------------------------------------------------------------------------

def _n_spikes_for_cell(t: dict, cell_number: int) -> int | None:
    """Handles all three repeat-trial spike-storage schemas found in this
    project (see 02_extract_repeats.ipynb section 1) plus the plain unique/
    flat case, so callers never need to know which one a given trial uses."""
    per_unit = t.get("per_unit")
    if per_unit is not None:
        if cell_number >= len(per_unit):
            return None
        return sum(len(r) for r in per_unit[cell_number]["spk_times"])
    spk = t.get("spk_times")
    if spk is None or cell_number >= len(spk):
        return None
    entry = spk[cell_number]
    if isinstance(entry, list):  # trial-level repeat schema: list of per-rep arrays
        return sum(len(r) for r in entry)
    return len(entry)  # flat array (unique regime, or flat/unsplit repeat)


# Regimes that are build-time bookkeeping, not analysis data -- excluded by
# default (see build_cell_trial_table's `valid_only`): 'diagnostic' (ph/sf/tf
# calibration trials -- real content, but a tuning/calibration trial, not one
# of this dataset's NAT/SBN/FF data types, 637 rows currently) and 'n/a'
# (INVALID: discarded/aborted recordings, e.g. c11awn060, which failed a
# trigger-count check and is absent from the lab's own quality spreadsheet,
# 7 rows currently). 'unknown' is also included defensively (a few
# classify_file() failure paths use it) even though it doesn't currently
# occur in the built dataset. Deliberately does NOT include
# 'unique_or_repeat' (Expt3's ff/bff/cff -- uncharacterized structure, but
# not junk -- 28 rows) which stays visible by default.
_BOOKKEEPING_REGIMES = {"diagnostic", "n/a", "unknown"}


def _frames_per_rep(experiment_name: str, code: str | None, n_reps: int | None) -> int | None:
    """Validated stimulus frames per repetition for a repeat trial, by
    (experiment, code) -- see repeat_blocks.py/CLAUDE.md for how each
    value was established. Confidence varies: Expt1 ff (960), Expt3 nr/wr,
    Expt4 wr/nr (600) and fff_hl16/ff/fff (1800), and Expt5's whole repeat
    family (600) are validated against real ground truth (MATLAB output or
    Expt5.xls firing rates); Expt2's nr/wr (250) is only partially
    validated (real episodic structure shown, not the full documented
    block structure). Expt3's `nr` code needs `n_reps` to disambiguate its
    long (5400, letters c/f, n_reps=24) vs short (600, letters d/e,
    n_reps=30) variants -- code alone doesn't distinguish them; returns
    None for its 3 irregular penetrations (g32/h31/h32), whose n_reps
    (256/28/28) match neither. Expt5 doesn't store `code` on repeat trials
    at all (a known gap, see CLAUDE.md) -- its whole repeat family uses
    600 uniformly regardless, so this doesn't need `code` for Expt5."""
    if experiment_name == "Expt1" and code == "ff":
        return 960
    if experiment_name == "Expt2":
        if code in ("ff020", "ff055"):
            return 960
        if code in ("nr015", "nr040", "wr020", "wr055"):
            return 250
    if experiment_name == "Expt3":
        if code == "nr":
            if n_reps == 24:
                return 5400
            if n_reps == 30:
                return 600
            return None
        if code == "wr":
            return 600
    if experiment_name == "Expt4":
        if code in ("wr", "nr"):
            return 600
        if code in ("fff_hl16", "ff", "fff"):
            return 1800
    if experiment_name == "Expt5":
        return 600
    return None


def build_cell_trial_table(experiments: list[str] | None = None, root: str | None = None,
                            valid_only: bool = True) -> pd.DataFrame:
    """One row per (experiment_number, penetration_number, cell_number,
    trial_key) across the dataset -- the flat index `find_cells()`/
    `cell_info()` filter. Loads every penetration's `.pkl` once (cheap,
    files are a few MB each since the HDF5 stimulus-library refactor).

    `valid_only=True` (default) excludes build-time bookkeeping trials --
    `regime in {'diagnostic', 'n/a', 'unknown'}` (calibration/tuning trials,
    and discarded/aborted recordings) -- these would otherwise show up in
    every `find_cells()`/`cell_info()` result despite never being useful for
    analysis. Pass `valid_only=False` to see everything, e.g. for build-time
    debugging."""
    sessions = _cached_session_table(root)
    rows = []
    for s in sessions.itertuples():
        if experiments is not None and s.experiment_name not in experiments:
            continue
        path = os.path.join(_dataset_dir(s.experiment_name, root), f"{s.penetration_name}.pkl")
        with open(path, "rb") as f:
            ds = pickle.load(f)
        units = ds["units"]
        for trial_key, t in ds["trials"].items():
            if valid_only and t.get("regime") in _BOOKKEEPING_REGIMES:
                continue
            stim_ref = t.get("stim_ref")
            # n_reps/frames_per_rep: only meaningful for regime='repeat'.
            # n_reps comes directly off the trial where already stored
            # ('per_unit'/'trial_level' shapes); 'flat'-shape trials
            # (Expt1 ff, Expt2 nr/wr) don't store it, so it's derived from
            # len(frame_times) once frames_per_rep is known below.
            rep_n_reps = t.get("n_reps") if t.get("regime") == "repeat" else None
            rep_frames_per_rep = (
                _frames_per_rep(s.experiment_name, t.get("code"), rep_n_reps)
                if t.get("regime") == "repeat" else None
            )
            if rep_n_reps is None and rep_frames_per_rep and t.get("frame_times") is not None:
                rep_n_reps = len(t["frame_times"]) // rep_frames_per_rep
            for cell_number, u in enumerate(units):
                rows.append({
                    "experiment_number": s.experiment_number,
                    "penetration_number": s.penetration_number,
                    "cell_number": cell_number,
                    "experiment_name": s.experiment_name,
                    "penetration_name": s.penetration_name,
                    "cell_name": u["name"],
                    "trial_key": trial_key,
                    "modality": t.get("modality"),
                    "regime": t.get("regime"),
                    "code": t.get("code"),
                    "contrast": t.get("contrast"),
                    "contrast_rms": t.get("contrast_rms"),
                    "xy_class": u.get("xy_class"),
                    "xy_class_source": u.get("xy_class_source"),
                    "xy_class_log_f2f1": u.get("xy_class_log_f2f1"),
                    "quality_mean": u.get("quality_mean"),
                    "n_spikes": _n_spikes_for_cell(t, cell_number),
                    "has_stim": t.get("stim") is not None or stim_ref is not None,
                    # stim_family/stim_start/stim_stop: where this trial's content
                    # sits in the shared WN/NAT family sequence (None for FF trials,
                    # whose small stim is embedded directly, not referenced). Two
                    # trials with the same family+start are frame-for-frame in
                    # register already -- see load_trial()'s docstring for how to
                    # use this for cross-recording alignment.
                    "stim_family": stim_ref["family"] if stim_ref else None,
                    "stim_start": stim_ref["start"] if stim_ref else None,
                    "stim_stop": stim_ref["stop"] if stim_ref else None,
                    # n_frames: trial length in frames. For regime='repeat':
                    # ALWAYS the TOTAL elapsed stimulus frames across every
                    # repetition (n_reps * frames_per_rep), checked first and
                    # deliberately never falls through to stim_ref/frame_times
                    # below -- a repeat's stim_ref (where set at all, currently
                    # only Expt3's standard-pattern nr) describes one
                    # repetition's distinct content span, which is a different
                    # number from the trial's total length and would silently
                    # give the wrong answer here if checked first (caught
                    # exactly this way while adding frames_per_rep -- a11cnr's
                    # n_frames came out 5400, not 24*5400=129600). Use
                    # frames_per_rep directly if you want that per-repetition
                    # content length instead. For regime='unique': from
                    # stim_ref where set (identical to stim_stop-stim_start,
                    # just here for direct use without subtracting), else from
                    # a raw frame_times array where the trial has one. None
                    # wherever the underlying count isn't known/derivable (e.g.
                    # Expt3's 3 irregular nr penetrations).
                    "n_frames": (
                        (rep_n_reps * rep_frames_per_rep) if (rep_n_reps is not None and rep_frames_per_rep is not None)
                        else (stim_ref["stop"] - stim_ref["start"]) if stim_ref
                        else len(t["frame_times"]) if t.get("frame_times") is not None
                        else None
                    ),
                    # frames_per_rep/n_reps: only populated for regime='repeat'
                    # (None for unique trials) -- see _frames_per_rep's
                    # docstring for per-(experiment, code) provenance/confidence.
                    "frames_per_rep": rep_frames_per_rep,
                    "n_reps": rep_n_reps,
                })
    return pd.DataFrame(rows)


def find_cells(modality: str | None = None, regime: str | None = None, xy_class: str | None = None,
                min_spikes: int = 1, valid_only: bool = True, frame_thresh: int | None = None,
                table: pd.DataFrame | None = None, root: str | None = None, **filters) -> pd.DataFrame:
    """(1) "What cells have this type of data?" `modality`/`regime` are the
    two axes this dataset is organized along (`modality`: `'NAT'`/`'SBN'`/
    `'FF'`; `regime`: `'unique'` for long single-pass sequences, `'repeat'`
    for short repeated ones). `xy_class` optionally subsets to `'x'`/`'y'`
    cells; left as the default `None`, cells are returned regardless of
    X/Y classification. Any other column in `build_cell_trial_table()` can
    be filtered too via `column=value` kwargs (e.g. `contrast='HC'`).
    `valid_only` (default `True`) excludes build-time bookkeeping trials
    (calibration/tuning `'diagnostic'` trials, discarded/aborted `'n/a'`
    recordings) -- only takes effect when this function builds its own
    table (i.e. `table=None`); a `table=` you pass in is used as-is.

    `frame_thresh` (2026-08, soft-coded add-on -- see CLAUDE.md "SBN trial
    consistency" for why this isn't baked into `regime`/`code` yet):
    restricts to trials whose stim content spans at least this many frames
    (`stim_stop - stim_start >= frame_thresh`), for filtering out short
    calibration/spot-check sequences (e.g. Expt2's `wntest`, 6000 frames)
    when you want genuine long single-pass recordings -- most useful with
    `modality='SBN'` (this ambiguity doesn't arise for NAT). Trials without
    a `stim_ref` are excluded automatically (`NaN >= frame_thresh` is
    `False`). Also excludes `code == 'wntrio'` -- Expt4's synthetic
    multi-recording pooling has the SAME nominal duration as a real single
    pass but is known-unreliable for RF localization (see CLAUDE.md), so a
    length check alone can't separate it out; this is a deliberate
    exclusion, not a side effect. Does NOT distinguish REDUNDANT vs
    COMPLEMENTARY multiple passes within one penetration (e.g. Expt5's
    `c11`, where some passes repeat the same content and others cover
    different segments of the family) -- check the result's `stim_start`/
    `stim_stop` columns directly if that distinction matters; the
    information is already there, just not adjudicated by this filter.

    Returns one row per matching (cell, trial) -- includes `trial_key`, so
    the result can be handed straight to `load_cell()` with no further
    lookup. Pass a pre-built `table=` to avoid rescanning the dataset for
    repeated searches."""
    df = table if table is not None else build_cell_trial_table(root=root, valid_only=valid_only)
    mask = pd.Series(True, index=df.index)
    if modality is not None:
        mask &= df.modality == modality
    if regime is not None:
        mask &= df.regime == regime
    if xy_class is not None:
        mask &= df.xy_class == xy_class
    if frame_thresh is not None:
        mask &= (df.stim_stop - df.stim_start) >= frame_thresh
        mask &= df.code != "wntrio"
    for col, val in filters.items():
        mask &= df[col] == val
    mask &= df.n_spikes.fillna(0) >= min_spikes
    return df[mask].reset_index(drop=True)


def cell_info(experiment_number: int, penetration_number: int, cell_number: int,
              valid_only: bool = True, table: pd.DataFrame | None = None, root: str | None = None) -> pd.DataFrame:
    """(2) "What data is available for this cell?" -- every trial this cell
    appears in, with modality/regime/contrast/n_spikes, indexed purely by
    number (or see `number_for()` if starting from a name). `valid_only`
    (default `True`) excludes build-time bookkeeping trials (calibration/
    tuning `'diagnostic'` trials, discarded/aborted `'n/a'` recordings) --
    only takes effect when this function builds its own table (`table=None`);
    a `table=` you pass in is used as-is."""
    df = table if table is not None else build_cell_trial_table(root=root, valid_only=valid_only)
    m = (
        (df.experiment_number == experiment_number)
        & (df.penetration_number == penetration_number)
        & (df.cell_number == cell_number)
    )
    return df[m].reset_index(drop=True)


def load_cell(experiment_number: int, penetration_number: int, cell_number: int, trial_key: str,
              lib: StimLibraryH5 | None = None, root: str | None = None) -> dict:
    """Load one cell's data for one trial -- `trial_key` normally comes
    straight from a `find_cells()`/`cell_info()` row, not typed by hand.
    Same content as `load_trial()` but pre-sliced to this one cell instead
    of every unit in the penetration."""
    trial = load_trial(experiment_number, penetration_number, trial_key, lib=lib, root=root)
    return {
        "stim": trial["stim"],
        "stim_ref": trial["stim_ref"],  # {"family", "start", "stop"} -- see load_trial()'s docstring on using this for alignment
        "frame_times": trial["frame_times"],
        "spk_times": trial["spk_times"][cell_number] if trial["spk_times"] is not None else None,
        "per_unit": trial["per_unit"][cell_number] if trial["per_unit"] is not None else None,
        "cell_name": trial["unit_names"][cell_number],
        "xy_class": trial["xy_class"][cell_number],
        "xy_class_source": trial["xy_class_source"][cell_number],
        "xy_class_log_f2f1": trial["xy_class_log_f2f1"][cell_number],
        "contrast": trial["contrast"],
        "contrast_rms": trial["contrast_rms"],
        "regime": trial["regime"],
        "modality": trial["modality"],
        "notes": trial["notes"],
    }


# ---------------------------------------------------------------------------
# 4. Repeat trials
# ---------------------------------------------------------------------------

# _repeat_boundaries/_build_repeat_trial_blocked: intentional duplicates of
# repeat_blocks.py's repeat_boundaries()/build_repeat_trial_blocked() (same
# logic, private names here) -- NOT imported from there. User design
# decision (2026-08): the read/interpret side of this project should be
# static and independent of the build side once the dataset is built, not
# sharing live code with it. If the build-side splitting logic ever needs
# to change (e.g. a future rebuild), update both copies deliberately at
# that point -- until then this is deliberate duplication, not drift. See
# CLAUDE.md for the full rationale. repeat_blocks.py itself is untouched
# and still used by build_dataset.py/build_dataset_expt3.py/
# build_dataset_expt4.py exactly as before.

def _repeat_boundaries(frame_times, n_reps, trig_per_rep):
    """Real per-repeat time boundaries (relative to the trial start), from
    actual trigger timestamps -- NOT a nominal rep*duration grid (a nominal
    fixed-dt grid drifts substantially over a long recording, confirmed
    ~528ms drift by repeat 96 for an Expt5 file -- see CLAUDE.md).

    Returns (boundaries, t_start) where boundaries has length n_reps+1."""
    ft = np.asarray(frame_times)
    t_start = ft[0]
    n_trig = len(ft)
    boundary_idx = np.arange(n_reps + 1) * trig_per_rep
    boundary_idx[-1] = min(boundary_idx[-1], n_trig - 1)
    return ft[boundary_idx] - t_start, t_start


def _build_repeat_trial_blocked(frame_times, spk_times, n_reps=128, trig_per_rep=600,
                                 block_size=32, skip_first_of_block=True):
    """Split one unit's spike times into per-repeat arrays with HC/LC
    contrast labels, using a configurable contrast-block convention.
    Defaults are the Expt3/Expt4 convention (block_size=32,
    skip_first_of_block=True); pass block_size=16, skip_first_of_block=False
    for Expt5. Returns {'spk_times': list[n_reps] of 1D arrays (each
    repeat's spikes relative to that repeat's own start), 'contrast_per_rep':
    array[n_reps] of 'HC'/'LC', 'valid_rep': array[n_reps] bool (False for
    discarded warm-up reps), 'rep_durations': array[n_reps], 'n_reps'}."""
    rb, t_start = _repeat_boundaries(frame_times, n_reps, trig_per_rep)
    s = np.asarray(spk_times) - t_start

    reps, contrast, valid = [], [], []
    for r in range(n_reps):
        t0, t1 = rb[r], rb[r + 1]
        reps.append(s[(s >= t0) & (s < t1)] - t0)
        block = r // block_size
        contrast.append("HC" if block % 2 == 0 else "LC")
        valid.append(not (skip_first_of_block and (r % block_size == 0)))

    return {
        "spk_times": reps,
        "contrast_per_rep": np.array(contrast),
        "valid_rep": np.array(valid),
        "rep_durations": np.diff(rb),
        "n_reps": n_reps,
    }


# Validated trig_per_rep values for 'flat'-shape repeat trials, keyed by
# (experiment_name, code) -- see 02_extract_repeats.ipynb section 2 for the
# derivation of each. Expt2's ff020/ff055 are already per_unit-shaped at
# build time (not flat), so they don't need an entry here.
#
# BUG FIXED (2026-08): Expt2's nr/wr keys used to be the bare codes "nr"/
# "wr", which never matched -- Expt2's real stored codes carry a contrast
# suffix (nr015/nr040/wr020/wr055, confirmed directly against the built
# data). load_repeat_trial()/load_repeat_cell() on any Expt2 nr/wr trial
# raised "trig_per_rep unknown" unconditionally until this was fixed --
# caught while adding frames_per_rep to build_cell_trial_table(), which
# needed the same lookup and exposed the same mismatch.
_FLAT_TRIG_PER_REP: dict[tuple[str, str], int] = {
    ("Expt1", "ff"): 960,
    ("Expt2", "nr015"): 250,
    ("Expt2", "nr040"): 250,
    ("Expt2", "wr020"): 250,
    ("Expt2", "wr055"): 250,
}


def classify_repeat_shape(t: dict) -> str:
    """Which of the three on-disk repeat-trial storage schemas `t` uses:

    - `'per_unit'`: `t['per_unit']` is a list of one dict per unit, each with
      its own `spk_times`/`contrast_per_rep`/`valid_rep`/`rep_durations`.
    - `'trial_level'`: `t['spk_times']` is directly a list[unit][rep] of
      arrays, with `contrast_per_rep`/`rep_durations` sitting once at the
      trial level (not duplicated per unit).
    - `'flat'`: just `frame_times` + `spk_times` (one unsplit array per
      unit) -- repeats haven't been split at all and must be reconstructed
      on the fly (see `load_repeat_trial`).

    See `02_extract_repeats.ipynb` section 1 for the original survey this
    generalizes."""
    if t.get("per_unit") is not None:
        return "per_unit"
    spk = t.get("spk_times")
    if t.get("contrast_per_rep") is not None and isinstance(spk, list) and len(spk) and isinstance(spk[0], list):
        return "trial_level"
    return "flat"


def load_repeat_trial(experiment_number: int, penetration_number: int, trial_key: str,
                       trig_per_rep: int | None = None, block_size: int | None = None,
                       skip_first_of_block: bool = False, lib: StimLibraryH5 | None = None,
                       root: str | None = None) -> dict:
    """Load one repeat trial (`regime == 'repeat'`) and normalize it to a
    uniform, PSTH-ready shape regardless of which of the three on-disk
    storage schemas it uses (see `classify_repeat_shape`'s docstring).

    Returns `{'shape', 'per_unit': [...], 'unit_names', 'stim', 'stim_ref',
    'xy_class', 'xy_class_source', 'xy_class_log_f2f1', 'modality', 'code',
    'contrast', 'contrast_rms', 'regime', 'notes'}`, where `per_unit[i]` =
    `{'spk_times': [rep0_array, rep1_array, ...], 'contrast_per_rep',
    'rep_durations', 'valid_rep'}`.

    For `'flat'`-shape trials, `trig_per_rep` is required to reconstruct
    repeat boundaries -- auto-looked-up from `_FLAT_TRIG_PER_REP` by
    `(experiment_name, code)` if not passed explicitly; raises if unknown
    and not supplied (no silent guessing). `block_size`/`skip_first_of_block`
    only matter for `'flat'` trials (default: one block spanning all reps,
    i.e. no contrast split) -- `'per_unit'`/`'trial_level'` trials already
    have their contrast structure baked in at build time and ignore these.

    `stim`/`stim_ref` resolved exactly like `load_trial()` (via `lib` if a
    `stim_ref` is present, else the embedded `stim`). Real for Expt3's NAT
    `nr` repeats on standard-pattern penetrations (**unverified** -- see the
    trial's own `notes`, and CLAUDE.md) and for Expt1's `ff`/Expt2's
    `ff020`/`ff055` (small embedded FF stim); `None` for every other
    NAT/SBN repeat in this dataset -- content genuinely not identified."""
    experiment, penetration = name_for(experiment_number, penetration_number, root)
    path = os.path.join(_dataset_dir(experiment, root), f"{penetration}.pkl")
    with open(path, "rb") as f:
        ds = pickle.load(f)
    t = ds["trials"][trial_key]
    if t.get("regime") != "repeat":
        raise ValueError(f"{trial_key!r} is regime={t.get('regime')!r}, not 'repeat'")

    shape = classify_repeat_shape(t)
    if shape == "per_unit":
        per_unit = [
            {
                "spk_times": u["spk_times"],
                "contrast_per_rep": list(u["contrast_per_rep"]),
                "rep_durations": list(u["rep_durations"]),
                "valid_rep": list(u.get("valid_rep", [True] * u["n_reps"])),
            }
            for u in t["per_unit"]
        ]
    elif shape == "trial_level":
        contrast = list(t["contrast_per_rep"])
        durations = list(t["rep_durations"])
        per_unit = [
            {"spk_times": unit_reps, "contrast_per_rep": contrast, "rep_durations": durations, "valid_rep": [True] * len(contrast)}
            for unit_reps in t["spk_times"]
        ]
    else:  # flat
        if trig_per_rep is None:
            trig_per_rep = _FLAT_TRIG_PER_REP.get((experiment, t.get("code")))
        if trig_per_rep is None:
            raise ValueError(
                f"trig_per_rep unknown for {experiment}/{t.get('code')!r} -- pass it explicitly "
                f"(see 02_extract_repeats.ipynb section 2 for how known values were validated)"
            )
        ft = t["frame_times"]
        n_reps = len(ft) // trig_per_rep
        bs = block_size or n_reps
        per_unit = []
        for spk in t["spk_times"]:
            r = _build_repeat_trial_blocked(ft, spk, n_reps=n_reps, trig_per_rep=trig_per_rep,
                                             block_size=bs, skip_first_of_block=skip_first_of_block)
            per_unit.append({
                "spk_times": r["spk_times"],
                "contrast_per_rep": list(r["contrast_per_rep"]),
                "rep_durations": list(r["rep_durations"]),
                "valid_rep": list(r["valid_rep"]),
            })

    owns_lib = lib is None
    if owns_lib:
        lib = StimLibraryH5(root=root)
    try:
        stim = lib.resolve(t.get("stim_ref")) if t.get("stim_ref") is not None else t.get("stim")
    finally:
        if owns_lib:
            lib.close()

    return {
        "shape": shape,
        "per_unit": per_unit,
        "unit_names": [u["name"] for u in ds["units"]],
        "stim": stim,
        "stim_ref": t.get("stim_ref"),
        "xy_class": [u.get("xy_class") for u in ds["units"]],
        "xy_class_source": [u.get("xy_class_source") for u in ds["units"]],
        "xy_class_log_f2f1": [u.get("xy_class_log_f2f1") for u in ds["units"]],
        "modality": t.get("modality"),
        "code": t.get("code"),
        "contrast": t.get("contrast"),
        "contrast_rms": t.get("contrast_rms"),
        "regime": t.get("regime"),
        "notes": t.get("notes", []),
    }


def load_repeat_cell(experiment_number: int, penetration_number: int, cell_number: int, trial_key: str,
                      trig_per_rep: int | None = None, block_size: int | None = None,
                      skip_first_of_block: bool = False, lib: StimLibraryH5 | None = None,
                      root: str | None = None) -> dict:
    """Same as `load_repeat_trial` but pre-sliced to one cell -- mirrors
    `load_cell()`'s relationship to `load_trial()`. `trial_key` normally
    comes straight from a `find_cells(regime='repeat', experiment_number=...,
    penetration_number=..., cell_number=...)` row."""
    trial = load_repeat_trial(experiment_number, penetration_number, trial_key, trig_per_rep=trig_per_rep,
                               block_size=block_size, skip_first_of_block=skip_first_of_block, lib=lib, root=root)
    u = trial["per_unit"][cell_number]
    return {
        "spk_times": u["spk_times"],
        "contrast_per_rep": u["contrast_per_rep"],
        "rep_durations": u["rep_durations"],
        "valid_rep": u["valid_rep"],
        "cell_name": trial["unit_names"][cell_number],
        "stim": trial["stim"],
        "stim_ref": trial["stim_ref"],
        "xy_class": trial["xy_class"][cell_number],
        "xy_class_source": trial["xy_class_source"][cell_number],
        "xy_class_log_f2f1": trial["xy_class_log_f2f1"][cell_number],
        "modality": trial["modality"],
        "code": trial["code"],
        "contrast": trial["contrast"],
        "contrast_rms": trial["contrast_rms"],
        "regime": trial["regime"],
        "notes": trial["notes"],
    }


def compute_psth(repeat_cell: dict, binw: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """(bin_centers, rate_hz) from a `load_repeat_cell()`-style dict (or one
    entry of `load_repeat_trial()['per_unit']`). Averages over `valid_rep`
    reps only; bins span `[0, median(rep_durations))`."""
    trep = np.median(repeat_cell["rep_durations"])
    nbins = int(trep / binw)
    valid_reps = [s for s, v in zip(repeat_cell["spk_times"], repeat_cell["valid_rep"]) if v]
    counts = np.zeros(nbins)
    for s in valid_reps:
        c, _ = np.histogram(s, bins=nbins, range=(0, trep))
        counts += c
    rate = counts / max(len(valid_reps), 1) / binw
    bin_centers = (np.arange(nbins) + 0.5) * binw
    return bin_centers, rate
