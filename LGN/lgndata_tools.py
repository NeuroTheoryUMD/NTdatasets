"""Portable read API for the built LGN Stage-1 dataset.

This is the ONE file meant to travel with the *data*, not the build
process -- if you copy `derived/` (the per-penetration `.pkl` files +
`stim_library.h5`) somewhere else, this single module is all you need
alongside it, nothing else from `lgn_tools/`. It doesn't care where its
own code lives; only where the data lives, via `set_data_root()`/`root=`
(see below). The scripts that *build* that data (`build_dataset_exptN.py`
and everything else in this folder, including `repeat_blocks.py`) are a
separate, one-repo-only concern and don't need to travel anywhere.

**Self-contained**: this one file reads the released data; it does not import anything from the build side of `lgn_tools/`.

Three layers, merged into one file on purpose (previously
`session_index.py` + `cell_index.py` + `stim_lib.py` -- same functions,
just one import now instead of three):

1. **Numbered session access** (`build_session_table`, `name_for`,
   `number_for`). Raw experiment/penetration *names* remain the key inside
   every `.pkl` file and every raw recording -- but for someone just
   working through this dataset, a flat number is easier than remembering
   "c11" vs "d12" vs "e21". `experiment_number` is the experiment's label
   number (Expt1..Expt4 = 1..4). `penetration_number` is 0-indexed within its
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
   in one of two shapes depending on trial type (see `classify_repeat_shape`)
   -- these functions normalize across them so callers never need to know
   which one a given trial uses. `find_cells(regime='repeat', ...)` covers
   *finding* repeats for a given cell/condition; these functions cover
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
import warnings

import h5py
import numpy as np
import pandas as pd

_FILE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_NAMES = ["Expt1", "Expt2", "Expt3", "Expt4"]  # the released experiments; experiment_number = the digit

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
    return os.path.join(_resolve_root(root), "derived", experiment)


# ---------------------------------------------------------------------------
# 1. Numbered session access
# ---------------------------------------------------------------------------

_session_table_cache: dict[str, pd.DataFrame] = {}


def build_session_table(root: str | None = None) -> pd.DataFrame:
    """One row per penetration across the 4 released experiments. Columns:
    session_index (flat, 0..N-1), experiment_number (1-4), experiment_name,
    penetration_number (0..k-1 within its experiment), penetration_name,
    session_date."""
    rows = []
    session_index = 0
    for experiment_name in EXPERIMENT_NAMES:
        experiment_number = int(experiment_name[4:])
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

    def resolve(self, stim_ref: dict | None):
        """`{family, start, stop}` -> the array. A reference with `by_contrast` (`{letter: [start, stop]}`, the FF_LONG trials, whose H and L
        repetitions show different chunks) -> a dict `{letter: array}`."""
        if stim_ref is None:
            return None
        if stim_ref.get("by_contrast"):
            return {letter: self._slice(stim_ref["family"], a, b) for letter, (a, b) in stim_ref["by_contrast"].items()}
        return self._slice(stim_ref["family"], stim_ref["start"], stim_ref["stop"])

    def _slice(self, family: str, start: int, stop: int) -> np.ndarray:
        key = (family, int(start), int(stop))
        if key not in self._cache:
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
                "contrast_rms": _rms_scalar(t.get("contrast_rms")),
                "contrast_rms_levels": _rms_levels(t.get("contrast"), t.get("contrast_rms")),
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


def _timing_for_trial(ds: dict, trial_key: str):
    """Timing summary of the raw recording behind a trial (from `meta['timing_by_file']`, written by the 2026-10-22 build):
    {'refreshes_per_trigger', 'dt_nominal_s', 'dt_effective_s', 'ppm_vs_nominal', 'n_segments', 'pauses', ...}, or None for
    recordings that are not a regular refresh-locked trigger train (phase-grating blocks, m-sequence) and for synthetic pooled
    trials built from several recordings (look the files up in `meta['timing_by_file']` directly)."""
    tb = (ds.get("meta") or {}).get("timing_by_file") or {}
    if trial_key in tb:
        return tb[trial_key]
    return tb.get(trial_key.rsplit("_", 1)[0])


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
        # block_triggers: only for modality 'mseq' (m-sequence, `l2` files) -- the 16 raw
        # (start, stop) trigger times of its 8 blocks, relative to the trial's first
        # trigger. frame_times are NOMINAL within each block there; use block_triggers to
        # mask spikes in the ~0.89 s inter-block gaps (they are kept in spk_times).
        "block_triggers": t.get("block_triggers"),
        # timing (2026-10-22): fixed-period summary of this trial's recording -- frame_times are a regular grid at
        # dt_effective_s (= STIMRES x trigger refreshes x (1 + ppm)); see CLAUDE.md "Stimulus timing". May be None.
        "timing": _timing_for_trial(ds, trial_key),
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
        # Hand annotations baked in at build time (annotations/*.csv; None / '' / [] where an older build or Expt1 has none).
        # Per unit (index-aligned with unit_names): electrode, subletter, LGN_layer ('A'/'A1'/'C'; a property of the electrode),
        # unit_notes (free text), possible_duplicate_of (names of units that are probably the SAME neuron seen on another channel --
        # nothing is removed, see find_cells(exclude_duplicates=...)), shared_input_with (different cells that share a retinal input).
        "electrode": [u.get("electrode") for u in ds["units"]],
        "subletter": [u.get("subletter") for u in ds["units"]],
        "LGN_layer": [u.get("LGN_layer") for u in ds["units"]],
        "unit_notes": [u.get("unit_notes") or "" for u in ds["units"]],
        "possible_duplicate_of": [list(u.get("possible_duplicate_of") or []) for u in ds["units"]],
        "shared_input_with": [list(u.get("shared_input_with") or []) for u in ds["units"]],
        # response_quality (released Expt1 penetrations a13 and a14 only; {} elsewhere): the lab's hand rating of each cell per trial type, {trial code: {'rating': 0-3, 'flags': 'fun letters', 'raw': text}};
        # 3 = perfect, reduced by 1 for low firing rate (f), instability across trials (u), noise in HC (n); 'x' = no data. response_quality_note: the lab's remark on a bad cell.
        "response_quality": [dict(u.get("response_quality") or {}) for u in ds["units"]],
        # spike-sorting quality (the lab's 1-6 scale; released Expt3 and Expt4 only, None elsewhere): quality_mean over the unit's recordings, quality_by_trial = {lab's recording label: rating}
        # (labels as written in the lab sheets), quality = the rating for this trial's recording (matched through the sequence letter).
        "quality_mean": [u.get("quality_mean") for u in ds["units"]],
        "quality_by_trial": [dict(u.get("quality_by_trial") or {}) for u in ds["units"]],
        "quality": [_quality_for_file(u, ds["meta"]["penetration"], t.get("file_letter")) for u in ds["units"]],
        "response_quality_note": [u.get("response_quality_note") or "" for u in ds["units"]],
        # Per penetration: receptive-field eccentricity in degrees of visual angle relative to the area centralis (0, 0);
        # x positive = right, y positive = up (so x <= 0 for the left visual field). None where not annotated.
        "eccentricity_x_deg": (ds.get("meta") or {}).get("eccentricity_x_deg"),
        "eccentricity_y_deg": (ds.get("meta") or {}).get("eccentricity_y_deg"),
        "penetration_notes": (ds.get("meta") or {}).get("penetration_notes"),
        # Recording sequence (see release_schema.py): file_letter, trial_order (rank within the penetration; None where the order is not fully known), source_files, duration_s.
        "file_letter": t.get("file_letter"),
        "trial_order": t.get("trial_order"),
        "source_files": t.get("source_files"),
        "duration_s": t.get("duration_s"),
        "accepted_assumptions": list(t.get("accepted_assumptions") or []),   # plain sentences; [] where the trial rests on no unchecked assumption (see release_schema.ACCEPTED_ASSUMPTIONS)
        # contrast: the letters of the contrast levels present, ordered X > H > L ('H', 'L', 'X', or a mixture 'HL', 'XHL'; None = unknown);
        # contrast_rms: a float for one level, {letter: value} for a mixture (X = 1.00, H and L by modality: NAT 0.40 / 0.15, SBN and FF 0.55 / 0.20;
        # the three-level designs use H = 0.552). contrast_segments: only for the raw white-noise trio recordings -- the stimulus-frame range of each level.
        "contrast": t.get("contrast"),
        "contrast_segments": t.get("contrast_segments"),
        # coverage: only for the pooled `trio` trials -- the stimulus-frame intervals [start, stop) that have data at this contrast level (all three 6000-frame blocks in
        # most penetrations; fewer in Expt3/g11, which has no recording of the second rotation). Outside them the trial has no recording at this contrast.
        "coverage": t.get("coverage"),
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


def _rms_scalar(rms):
    return float(rms) if isinstance(rms, (int, float)) else float("nan")


def _rms_levels(contrast, rms) -> str:
    if rms is None or contrast is None:
        return ""
    if isinstance(rms, dict):
        return " ".join(f"{k}={v:g}" for k, v in rms.items() if v is not None)
    return f"{contrast}={float(rms):g}"


def _quality_for_file(u: dict, pen: str, file_letter) -> float | None:
    """The lab's spike-sorting quality (1-6 scale; released Expt3 and Expt4 only) of unit `u` in the recording with this sequence letter. The lab sheets label recordings
    inconsistently (with or without the penetration name, a few typos such as 'd12dwn' for d12dwr), but each penetration has one recording per letter, so the letter is the join key."""
    if file_letter is None:
        return None
    for k, v in (u.get("quality_by_trial") or {}).items():
        kk = k[len(pen):] if k.startswith(pen) else k
        if kk[:1] == file_letter:
            return v
    return None


def _frames_per_rep(experiment_name: str, code: str | None, n_reps: int | None) -> int | None:
    """Trigger-resolution frames per repetition of a repeat trial, by (released experiment label, code): 960 for Expt1's `ff020`/`ff055`; 250 for
    Expt1's `nr`/`wr` blocks; Expt2's `nr` needs `n_reps` to tell its long variant (5400 frames, 24 reps) from the short one (600, 30 reps) and its
    build-time-pooled `nr_pooled` trials are 600; Expt3's `wr`/`nr`/`wr1short`/`fff_btrio` are 600 and `fff_hl16`/`ff`/`fff` 1800; Expt4's whole repeat
    family is 600 (except the binary full-field files, 7195 and 120, which store `trig_per_rep` in the trial and that value is used first). None where the structure is not established.
    (A trigger = 2 display refreshes; the 960-frame `ff` repeats have one frame per refresh.)"""
    if experiment_name == "Expt1":
        if code in ("ff020", "ff055"):
            return 960
        if code in ("nr015", "nr040", "wr020", "wr055"):
            return 250
    if experiment_name == "Expt2":
        if code == "nr":
            if n_reps == 24:
                return 5400
            if n_reps == 30:
                return 600
            return None
        if code == "wr":
            return 600
        if code == "nr_pooled":  # build-time-pooled short-clip repeat, see build_dataset_expt3.py
            return 600
    if experiment_name == "Expt3":
        if code in ("wr", "nr", "wr1short"):
            return 600
        if code in ("fff_hl16", "ff", "fff"):
            return 1800
        if code == "fff_btrio":
            return 600
    if experiment_name == "Expt4":
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
        # Hand annotations (annotations/*.csv, baked in at build time): same for every trial of a cell. The two list-valued
        # fields are stored as comma-joined strings ('' = none) so the table stays hashable/filterable; `duplicate_redundant`
        # is True for every member of a duplicate group except the one with the LOWEST cell_number (what exclude_duplicates drops).
        meta = ds.get("meta") or {}
        cell_of = {u["name"]: i for i, u in enumerate(units)}
        ann = []
        for i, u in enumerate(units):
            dups = list(u.get("possible_duplicate_of") or [])
            ann.append({
                "electrode": u.get("electrode"),
                "subletter": u.get("subletter"),
                "LGN_layer": u.get("LGN_layer"),
                "eccentricity_x_deg": meta.get("eccentricity_x_deg"),
                "eccentricity_y_deg": meta.get("eccentricity_y_deg"),
                "unit_notes": u.get("unit_notes") or "",
                "possible_duplicate_of": ", ".join(dups),
                "duplicate_redundant": any(cell_of.get(d, i) < i for d in dups),
                "shared_input_with": ", ".join(u.get("shared_input_with") or []),
                "response_quality_note": u.get("response_quality_note") or "",
            })
        for trial_key, t in ds["trials"].items():
            if valid_only and t.get("regime") in _BOOKKEEPING_REGIMES:
                continue
            stim_ref = t.get("stim_ref")
            # n_reps/frames_per_rep: only meaningful for regime='repeat'.
            # n_reps comes directly off the trial where already stored
            # ('per_unit'/'trial_level' shapes); where a trial doesn't store it, it is
            # derived from len(frame_times) once frames_per_rep is known below.
            rep_n_reps = t.get("n_reps") if t.get("regime") == "repeat" else None
            rep_frames_per_rep = (
                (t.get("trig_per_rep") or _frames_per_rep(s.experiment_name, t.get("code"), rep_n_reps))   # a stored per-trial value wins over the per-code table
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
                    # file_letter / trial_order: position of the recording in the penetration's recording sequence (the letter in the file name; trial_order is the 0-based rank, set only
                    # where every recording of the penetration has a letter); source_files: the recording file(s) the trial's data come from; duration_s: stimulus presentation time of
                    # the trial's own content (restart pauses excluded). None where not applicable / not known.
                    "file_letter": t.get("file_letter"),
                    "trial_order": t.get("trial_order"),
                    "source_files": ", ".join(t["source_files"]) if t.get("source_files") else None,
                    "duration_s": t.get("duration_s"),
                    "contrast": t.get("contrast"),
                    # contrast_rms: the RMS contrast as a float for a single-level trial (NaN for a mixture, which has no single value);
                    # contrast_rms_levels: the same information as a readable string for every trial, e.g. 'H=0.55' or 'H=0.55 L=0.2' / 'X=1 H=0.552 L=0.2'
                    "contrast_rms": _rms_scalar(t.get("contrast_rms")),
                    "contrast_rms_levels": _rms_levels(t.get("contrast"), t.get("contrast_rms")),
                    "xy_class": u.get("xy_class"),
                    "xy_class_source": u.get("xy_class_source"),
                    "xy_class_log_f2f1": u.get("xy_class_log_f2f1"),
                    # quality_mean: mean over the unit's recordings of the lab's spike-sorting quality (1-6; released Expt3, Expt4); quality: that rating for THIS trial's recording
                    "quality_mean": u.get("quality_mean"),
                    "quality": _quality_for_file(u, s.penetration_name, t.get("file_letter")),
                    # response_quality / response_quality_flags: the lab's hand rating (0-3; flags f/u/n) of this cell for this trial's stimulus code (released Expt1 a13, a14 only; NaN / '' elsewhere)
                    "response_quality": (u.get("response_quality") or {}).get(t.get("code"), {}).get("rating"),
                    "response_quality_flags": (u.get("response_quality") or {}).get(t.get("code"), {}).get("flags", ""),
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
                    **ann[cell_number],
                })
    return pd.DataFrame(rows)


_CONTRAST_ORDER = "XHL"
_CONTRAST_ALIAS = {"HC": "H", "LC": "L", "XC": "X"}


def find_cells(modality: str | None = None, regime: str | None = None, xy_class: str | None = None,
                contrast: str | None = None, contrast_exact: bool = False,
                min_spikes: int = 1, valid_only: bool = True, frame_thresh: int | None = None,
                exclude_duplicates: bool = False,
                table: pd.DataFrame | None = None, root: str | None = None, **filters) -> pd.DataFrame:
    """(1) "What cells have this type of data?" `modality`/`regime` are the
    two axes this dataset is organized along (`modality`: `'NAT'`/`'SBN'`/
    `'FF'`/`'mseq'` [16x16 m-sequence mapping, Expt2's `l2` files]/
    `'gray'` [constant mean-gray screen, Expt2's 90 s `mn` trials]; matching is
    case-insensitive, so `'MSEQ'`/`'nat'` work too;
    `regime`: `'unique'` for long single-pass sequences of one real recording, `'repeat'`
    for short repeated ones, `'trio'` for Expt3's pooled three-contrast white-noise trials, which combine three real recordings).
    `xy_class` optionally subsets to `'x'`/`'y'`
    cells; left as the default `None`, cells are returned regardless of
    X/Y classification. Any other column in `build_cell_trial_table()` can
    be filtered too via `column=value` kwargs (e.g. `LGN_layer='A1'`).

    `contrast` selects trials by contrast letter ('X' = RMS 1.00, 'H' = high, 'L' = low; the trial-level `contrast` column is the string of the
    levels present, e.g. 'H', 'L', 'HL' for repeats whose reps alternate H/L, 'XHL' for the three-level designs; None = unknown). By default a trial matches
    when it CONTAINS every letter you give (`contrast='H'` also returns 'HL' and 'XHL' trials; `contrast='HL'` returns trials containing both);
    `contrast_exact=True` requires equality ('H' only). The old spellings 'HC'/'LC'/'XC' are accepted as aliases of 'H'/'L'/'X'.
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
    `False`). Pooled `regime='trio'` trials have the same nominal duration as a real
    single pass but combine three real recordings; pass `regime='unique'` to leave them out. Does NOT distinguish REDUNDANT vs
    COMPLEMENTARY multiple passes within one penetration (e.g. Expt5's
    `c11`, where some passes repeat the same content and others cover
    different segments of the family) -- check the result's `stim_start`/
    `stim_stop` columns directly if that distinction matters; the
    information is already there, just not adjudicated by this filter.

    `exclude_duplicates` (default `False`: every unit is returned, nothing is dropped silently). A few pairs/groups of units on
    different channels are probably the SAME neuron recorded twice (cross-correlogram + identical receptive fields; flagged in the
    `possible_duplicate_of` column with the names of the other member(s)). If the result contains any such unit and
    `exclude_duplicates` is `False`, a `UserWarning` says so. `exclude_duplicates=True` keeps ONE unit per group -- the one with
    the lowest `cell_number` -- and silences the warning. (The `shared_input_with` column marks DIFFERENT cells that share a
    retinal input; those are never excluded and never warned about.) Other per-cell annotation columns in the result: `electrode`,
    `subletter`, `LGN_layer` ('A'/'A1'/'C'), `eccentricity_x_deg`/`eccentricity_y_deg` (degrees from the area centralis; x < 0 is the
    left visual field), `unit_notes`.

    Returns one row per matching (cell, trial) -- includes `trial_key`, so
    the result can be handed straight to `load_cell()` with no further
    lookup. Pass a pre-built `table=` to avoid rescanning the dataset for
    repeated searches."""
    df = table if table is not None else build_cell_trial_table(root=root, valid_only=valid_only)
    mask = pd.Series(True, index=df.index)
    if modality is not None:
        mask &= df.modality.str.lower() == modality.lower()
    if regime is not None:
        mask &= df.regime == regime
    if xy_class is not None:
        mask &= df.xy_class == xy_class
    if contrast is not None:
        want = "".join(_CONTRAST_ALIAS.get(contrast, contrast))
        have = df.contrast.fillna("").astype(str)
        mask &= (have == "".join(sorted(want, key=_CONTRAST_ORDER.index))) if contrast_exact else have.apply(lambda c: bool(c) and set(want) <= set(c))
    if frame_thresh is not None:
        mask &= (df.stim_stop - df.stim_start) >= frame_thresh
    for col, val in filters.items():
        mask &= df[col] == val
    mask &= df.n_spikes.fillna(0) >= min_spikes
    if exclude_duplicates and "duplicate_redundant" in df:
        mask &= ~df.duplicate_redundant.fillna(False).astype(bool)
    out = df[mask].reset_index(drop=True)
    if not exclude_duplicates and "possible_duplicate_of" in out and (out.possible_duplicate_of != "").any():
        flagged = out.loc[out.possible_duplicate_of != "", ["experiment_number", "penetration_number", "cell_number"]].drop_duplicates()
        n_cells = len(out[["experiment_number", "penetration_number", "cell_number"]].drop_duplicates())
        warnings.warn(
            f"{len(flagged)} of the {n_cells} cells returned are flagged in `possible_duplicate_of` (probably the same neuron recorded on "
            "more than one channel); all are included. Pass exclude_duplicates=True to keep one unit per group.",
            UserWarning, stacklevel=2,
        )
    return out


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
        "block_triggers": trial["block_triggers"],
        "timing": trial["timing"],
        "spk_times": trial["spk_times"][cell_number] if trial["spk_times"] is not None else None,
        "per_unit": trial["per_unit"][cell_number] if trial["per_unit"] is not None else None,
        "cell_name": trial["unit_names"][cell_number],
        "xy_class": trial["xy_class"][cell_number],
        "xy_class_source": trial["xy_class_source"][cell_number],
        "xy_class_log_f2f1": trial["xy_class_log_f2f1"][cell_number],
        "electrode": trial["electrode"][cell_number],
        "subletter": trial["subletter"][cell_number],
        "LGN_layer": trial["LGN_layer"][cell_number],
        "unit_notes": trial["unit_notes"][cell_number],
        "possible_duplicate_of": trial["possible_duplicate_of"][cell_number],
        "shared_input_with": trial["shared_input_with"][cell_number],
        "response_quality": trial["response_quality"][cell_number],
        "quality_mean": trial["quality_mean"][cell_number],
        "quality_by_trial": trial["quality_by_trial"][cell_number],
        "quality": trial["quality"][cell_number],
        "response_quality_note": trial["response_quality_note"][cell_number],
        "eccentricity_x_deg": trial["eccentricity_x_deg"],
        "eccentricity_y_deg": trial["eccentricity_y_deg"],
        "penetration_notes": trial["penetration_notes"],
        "file_letter": trial["file_letter"],
        "trial_order": trial["trial_order"],
        "source_files": trial["source_files"],
        "duration_s": trial["duration_s"],
        "accepted_assumptions": trial["accepted_assumptions"],
        "contrast": trial["contrast"],
        "contrast_segments": trial["contrast_segments"],
        "coverage": trial["coverage"],
        "contrast_rms": trial["contrast_rms"],
        "regime": trial["regime"],
        "modality": trial["modality"],
        "notes": trial["notes"],
    }


# ---------------------------------------------------------------------------
# 4. Repeat trials
# ---------------------------------------------------------------------------

# CRT refresh period in seconds (the lab's value, used throughout the original code as STIMRES).
# The display runs at this FIXED period; a trigger is 2 refreshes (16.678 ms) for almost all trial types.
_STIMRES_S = 8.33911056439e-3


def classify_repeat_shape(t: dict) -> str:
    """Which of the two on-disk repeat-trial storage schemas `t` uses:

    - `'per_unit'`: `t['per_unit']` is a list of one dict per unit, each with
      its own `spk_times`/`contrast_per_rep`/`valid_rep`/`rep_durations`.
    - `'trial_level'`: `t['spk_times']` is directly a list[unit][rep] of
      arrays, with `contrast_per_rep`/`rep_durations` sitting once at the
      trial level (not duplicated per unit).

    (A third, 'flat' schema -- unsplit spike trains -- existed only in the excluded pilot experiment and is gone.)"""
    if t.get("per_unit") is not None:
        return "per_unit"
    spk = t.get("spk_times")
    if t.get("contrast_per_rep") is not None and isinstance(spk, list) and len(spk) and isinstance(spk[0], list):
        return "trial_level"
    raise ValueError("not a recognised repeat-trial schema (neither per_unit nor trial_level)")


def load_repeat_trial(experiment_number: int, penetration_number: int, trial_key: str,
                       lib: StimLibraryH5 | None = None, root: str | None = None) -> dict:
    """Load one repeat trial (`regime == 'repeat'`) and normalize it to a
    uniform, PSTH-ready shape regardless of which of the two on-disk
    storage schemas it uses (see `classify_repeat_shape`'s docstring).

    Returns `{'shape', 'per_unit': [...], 'unit_names', 'stim', 'stim_ref',
    'dt', 'xy_class', 'xy_class_source', 'xy_class_log_f2f1', 'modality',
    'code', 'contrast', 'contrast_rms', 'regime', 'notes'}`, where
    `per_unit[i]` = `{'spk_times': [rep0_array, rep1_array, ...],
    'contrast_per_rep', 'rep_durations', 'valid_rep', 'rep_start_times'}`.
    `contrast` is the string of contrast letters present (`'H'`, `'L'`, `'HL'`, `'XHL'`, ...; `None` = unknown) and `contrast_per_rep[r]` the letter
    of repetition r ('unknown' where not known); `contrast_rms` is a float, or a `{letter: value}` dict for a mixture.

    `dt`: the stimulus frame period (s), a fixed number -- 2 x the CRT refresh STIMRES = 8.33911056439 ms for almost every repeat type
    (0.01667822 s; the 960-frame `ff020`/`ff055` repeats of Expt1 have 1 frame per refresh, 0.00833911 s). CORRECTED 2026-10-22 (it used to be
    mean(rep_durations)/frames_per_rep, which was inflated by the restart pauses). The actual frame spacing in this recording's own
    clock is `timing['dt_effective_s']` (= dt x (1 + 1-3 ppm)), see below.

    Repeat windows (2026-10-22 build): each rep is its CONTENT window, `rep_durations` = frames_per_rep x the trigger period
    (identical for all reps), counted from the rep's own first trigger, and `rep_start_times[r]` is the rep start relative to the trial's
    first trigger. The constant ~0.175 s restart pause that follows some reps (after every rep in Expt2 `nr` and Expt1 `ff`, after rep 64
    in the 128-rep Expt3/4 `wr`/`nr`, after every 32 reps in Expt4 `fff`, ...) is not part of any rep and its spikes are not stored; its
    length is diff(rep_start_times) - rep_durations. `timing` summarizes the fixed-period fit of the recording (None if not applicable).

    `stim`/`stim_ref` resolved exactly like `load_trial()` (via `lib` when a
    `stim_ref` is present; for a `by_contrast` reference -- the FF_LONG trials -- `stim` is a dict `{letter: samples}`); `None` where the content of a
    repeat trial is not identified."""
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
                "valid_rep": list(u["valid_rep"]) if u.get("valid_rep") is not None else [True] * len(u["spk_times"]),
                # 2026-10-22 build: rep start times (relative to the trial's first trigger); None for older/other schemas
                "rep_start_times": list(u["rep_start_times"]) if u.get("rep_start_times") is not None else None,
            }
            for u in t["per_unit"]
        ]
    elif shape == "trial_level":
        contrast = list(t["contrast_per_rep"])
        durations = list(t["rep_durations"])
        # valid_rep (2026-08): False for discarded warm-up reps (Expt5's
        # SBN/NAT wr/nr, 32-block/skip-first convention) -- falls back to
        # all-True for trial types that never had this (Expt5's fff, 16-block/
        # no-skip, unaffected by that fix).
        valid = list(t["valid_rep"]) if t.get("valid_rep") is not None else [True] * len(contrast)
        starts = list(t["rep_start_times"]) if t.get("rep_start_times") is not None else None
        per_unit = [
            {"spk_times": unit_reps, "contrast_per_rep": contrast, "rep_durations": durations, "valid_rep": valid,
             "rep_start_times": starts}
            for unit_reps in t["spk_times"]
        ]

    owns_lib = lib is None
    if owns_lib:
        lib = StimLibraryH5(root=root)
    try:
        stim = lib.resolve(t.get("stim_ref")) if t.get("stim_ref") is not None else t.get("stim")
    finally:
        if owns_lib:
            lib.close()

    # dt (2026-08): repeat trials carry no frame_times array at all (unlike
    # unique trials), so there was previously no way to recover even the
    # nominal stimulus frame period from load_repeat_trial's own output.
    # frames_per_rep is looked up via _frames_per_rep (the same validated table
    # build_cell_trial_table's own frames_per_rep column uses).
    # CORRECTED 2026-10-22: dt was the mean of rep_durations / frames_per_rep, but rep_durations are
    # start-to-start times of the quantized (25 us) trigger stamps AND include the constant ~0.175 s
    # pause where the stimulus movie restarted (after every rep in Expt3 `nr`, every rep in Expt2 `ff`,
    # after rep 64 in the 128-rep Expt4/5 files, ...), so dt came out up to 2% too large (Expt3 short
    # `nr` 0.01696 s). The display runs at a FIXED period: the CRT frame STIMRES = 8.33911056439 ms
    # (a trigger = 2 refreshes = 16.678 ms; the Expt1/Expt2 960-frame `ff` repeats have 1 frame per
    # trigger). Measured trigger periods match this to 1-3 ppm in every recording
    # (scratch_analysis_plots/stimulus_clock_check.py), so dt is now that nominal value.
    frames_per_rep = t.get("trig_per_rep") or _frames_per_rep(experiment, t.get("code"), t.get("n_reps"))   # a stored per-trial value wins over the per-code table
    dt = None
    if frames_per_rep:
        dt = _STIMRES_S * (1 if frames_per_rep == 960 else 2)

    return {
        "shape": shape,
        "per_unit": per_unit,
        "unit_names": [u["name"] for u in ds["units"]],
        "stim": stim,
        "stim_ref": t.get("stim_ref"),
        "timing": _timing_for_trial(ds, trial_key),
        "dt": dt,
        "xy_class": [u.get("xy_class") for u in ds["units"]],
        "xy_class_source": [u.get("xy_class_source") for u in ds["units"]],
        "xy_class_log_f2f1": [u.get("xy_class_log_f2f1") for u in ds["units"]],
        # per-unit annotations and per-penetration metadata: the same fields as load_trial() (see its comments)
        "electrode": [u.get("electrode") for u in ds["units"]],
        "subletter": [u.get("subletter") for u in ds["units"]],
        "LGN_layer": [u.get("LGN_layer") for u in ds["units"]],
        "unit_notes": [u.get("unit_notes") or "" for u in ds["units"]],
        "possible_duplicate_of": [list(u.get("possible_duplicate_of") or []) for u in ds["units"]],
        "shared_input_with": [list(u.get("shared_input_with") or []) for u in ds["units"]],
        "response_quality": [dict(u.get("response_quality") or {}) for u in ds["units"]],
        "response_quality_note": [u.get("response_quality_note") or "" for u in ds["units"]],
        "quality_mean": [u.get("quality_mean") for u in ds["units"]],
        "quality_by_trial": [dict(u.get("quality_by_trial") or {}) for u in ds["units"]],
        "quality": [_quality_for_file(u, ds["meta"]["penetration"], t.get("file_letter")) for u in ds["units"]],
        "eccentricity_x_deg": (ds.get("meta") or {}).get("eccentricity_x_deg"),
        "eccentricity_y_deg": (ds.get("meta") or {}).get("eccentricity_y_deg"),
        "penetration_notes": (ds.get("meta") or {}).get("penetration_notes"),
        "modality": t.get("modality"),
        "code": t.get("code"),
        "file_letter": t.get("file_letter"),
        "trial_order": t.get("trial_order"),
        "source_files": t.get("source_files"),
        "duration_s": t.get("duration_s"),
        "accepted_assumptions": list(t.get("accepted_assumptions") or []),   # plain sentences; [] where the trial rests on no unchecked assumption (see release_schema.ACCEPTED_ASSUMPTIONS)
        "contrast": t.get("contrast"),
        "contrast_rms": t.get("contrast_rms"),
        "regime": t.get("regime"),
        "notes": t.get("notes", []),
    }


def load_repeat_cell(experiment_number: int, penetration_number: int, cell_number: int, trial_key: str,
                      lib: StimLibraryH5 | None = None, root: str | None = None) -> dict:
    """Same as `load_repeat_trial` but pre-sliced to one cell -- mirrors
    `load_cell()`'s relationship to `load_trial()`. `trial_key` normally
    comes straight from a `find_cells(regime='repeat', experiment_number=...,
    penetration_number=..., cell_number=...)` row. Includes `dt` (see
    `load_repeat_trial`'s docstring) -- the stimulus frame period, since
    repeat trials otherwise carry no timing info beyond `rep_durations`."""
    trial = load_repeat_trial(experiment_number, penetration_number, trial_key, lib=lib, root=root)
    u = trial["per_unit"][cell_number]
    return {
        "spk_times": u["spk_times"],
        "contrast_per_rep": u["contrast_per_rep"],
        "rep_durations": u["rep_durations"],
        "valid_rep": u["valid_rep"],
        "rep_start_times": u["rep_start_times"],
        "timing": trial["timing"],
        "cell_name": trial["unit_names"][cell_number],
        "stim": trial["stim"],
        "stim_ref": trial["stim_ref"],
        "dt": trial["dt"],
        "xy_class": trial["xy_class"][cell_number],
        "xy_class_source": trial["xy_class_source"][cell_number],
        "xy_class_log_f2f1": trial["xy_class_log_f2f1"][cell_number],
        "electrode": trial["electrode"][cell_number],
        "subletter": trial["subletter"][cell_number],
        "LGN_layer": trial["LGN_layer"][cell_number],
        "unit_notes": trial["unit_notes"][cell_number],
        "possible_duplicate_of": trial["possible_duplicate_of"][cell_number],
        "shared_input_with": trial["shared_input_with"][cell_number],
        "response_quality": trial["response_quality"][cell_number],
        "response_quality_note": trial["response_quality_note"][cell_number],
        "quality_mean": trial["quality_mean"][cell_number],
        "quality_by_trial": trial["quality_by_trial"][cell_number],
        "quality": trial["quality"][cell_number],
        "eccentricity_x_deg": trial["eccentricity_x_deg"],
        "eccentricity_y_deg": trial["eccentricity_y_deg"],
        "penetration_notes": trial["penetration_notes"],
        "modality": trial["modality"],
        "code": trial["code"],
        "file_letter": trial["file_letter"],
        "trial_order": trial["trial_order"],
        "source_files": trial["source_files"],
        "duration_s": trial["duration_s"],
        "accepted_assumptions": trial["accepted_assumptions"],
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
