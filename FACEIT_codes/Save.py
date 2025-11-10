from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from pynwb.base import TimeSeries

# ----------------------------------------------------------------------------- #
# Logging
# ----------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")


class SaveHandler:
    """
    Handles saving Face-It results to compressed NPZ, NWB, and summary figures.

    Expected app_instance attributes:
      - save_path: str | Path (set by LoadData)
      - folder_path: str | Path (NPY image dir)  [optional]
      - video_path: str | Path (full path to video) [optional]
      - nwb_check(), pupil_check(), face_check()  [optional]
      - data arrays...
    """

    # ----------------------------- Public API -------------------------------- #
    def __init__(self, app_instance, base_path: Optional[Path] = None):
        self.app = app_instance
        self._override: Optional[Path] = Path(base_path) if base_path is not None else None
        self._run_id: Optional[str] = None

    @property
    def save_dir(self) -> Path:
        """Resolve base dir and ensure '<base>/faceIt' exists."""
        base = self._resolve_base_dir()
        return self._ensure_faceit_dir(base)

    # ----------------------------- Init / Save -------------------------------- #
    def _stamp_run(self) -> None:
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def init_save_data(self) -> None:
        """
        Prepare data (ensure existence + exact shapes), then save:
          - compressed NPZ
          - NWB
          - quick-look PNG figures
        """
        self._stamp_run()
        _ = self.save_dir

        # decide master T (frames)
        T = self._infer_length()

        # Enforce shapes for ALL fields (create missing, coerce wrong shapes)
        self._enforce_all(T)

        # Derived defaults that depend on others (after shapes are enforced)
        # If facemotion_without_grooming is all-NaN, fall back to motion_energy
        fmg = self._get_arr("facemotion_without_grooming")
        if fmg is not None and np.isnan(fmg).all():
            me = self._get_arr("motion_energy")
            if me is not None and len(me) == T:
                setattr(self.app, self._resolve_attr_target("facemotion_without_grooming"),
                        np.asarray(me, float))

        # Save artifacts
        try:
            self._save_npz()
        except Exception as e:
            logger.error("Failed to save NPZ: %s", e, exc_info=True)

        try:
            nwb_out = self.save_dir / "faceit.nwb"
            self._save_nwb(nwb_out)
        except Exception as e:
            logger.error("Failed to save NWB: %s", e, exc_info=True)

        try:
            self._save_figures()
        except Exception as e:
            logger.error("Failed to save figures: %s", e, exc_info=True)

    # ----------------------------- Path helpers ------------------------------- #
    def _resolve_base_dir(self) -> Path:
        """Priority: override -> save_path -> folder_path -> parent(video_path) -> HOME"""
        if self._override is not None:
            return self._override

        sp = getattr(self.app, "save_path", None)
        if isinstance(sp, (str, Path)) and str(sp).strip():
            p = Path(sp)
            if p.exists() and p.is_dir():
                return p

        fp = getattr(self.app, "folder_path", None)
        if isinstance(fp, (str, Path)) and str(fp).strip():
            p = Path(fp)
            if p.exists() and p.is_dir():
                return p

        vp = getattr(self.app, "video_path", None)
        if isinstance(vp, (str, Path)) and str(vp).strip():
            vpp = Path(vp)
            base = vpp.parent if vpp.suffix else vpp
            if base.exists():
                return base

        return Path.home()

    @staticmethod
    def _ensure_faceit_dir(base_path: Path) -> Path:
        out = Path(base_path) / "faceIt"
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ------------------------- Shape enforcement ------------------------------ #
    def _infer_length(self) -> int:
        """Choose a consistent primary length T for 1D signals."""
        pc = self._get_arr("pupil_center")
        if pc is not None and pc.size > 0:
            return int(pc.shape[0])
        me = self._get_arr("motion_energy")
        if me is not None and me.size > 0:
            return int(me.shape[0])
        logger.warning("Could not infer data length; defaulting to 1.")
        return 1

    def _target_shapes(self, T: int) -> Dict[str, Tuple[int, ...]]:
        """Single source of truth for every field's shape."""
        return {
            # pupil signals
            "pupil_center": (T,),
            "pupil_center_X": (T,),
            "pupil_center_y": (T,),
            "final_pupil_area": (T,),
            "pupil_dilation": (T,),
            "X_saccade_updated": (2, T),
            "Y_saccade_updated": (2, T),
            "pupil_distance_from_corner": (T,),
            "width": (T,),
            "height": (T,),
            "angle": (T,),
            "blinking_ids": (T,),

            # face signals
            "motion_energy": (T,),
            "facemotion_without_grooming": (T,),
            "grooming_ids": (T,),
            "grooming_thr": (1,),

            # frame references (fixed-size slots)
            "Face_frame": (4,),
            "Pupil_frame": (4,),
        }

    def _resolve_attr_target(self, attr: str) -> str:
        """
        Return the attribute name on `self.app` we should actually read/write.
        If `self.app.attr` exists but is callable (e.g. QWidget.width()),
        redirect to a safe alternate name like 'attr_arr'.
        """
        cur = getattr(self.app, attr, None)
        if callable(cur):
            return f"{attr}_arr"
        return attr

    def _ensure_exact(self, attr: str, target_shape: Tuple[int, ...], *, dtype=float) -> None:
        """
        Ensure self.app.<attr> exists (under a non-conflicting name) and has exactly target_shape.
        If missing => create with NaNs; if shape mismatch => pad/trim.
        """
        # Resolve to safe attribute (handles QWidget.width/height collisions)
        attr_target = self._resolve_attr_target(attr)
        cur = getattr(self.app, attr_target, None)

        # Treat callables or weird types as "missing"
        if (cur is None) or callable(cur):
            setattr(self.app, attr_target, np.full(target_shape, np.nan, dtype=dtype))
            return

        # Try to coerce to ndarray
        try:
            arr = np.asarray(cur, dtype=dtype)
        except Exception:
            setattr(self.app, attr_target, np.full(target_shape, np.nan, dtype=dtype))
            return

        if arr.shape == target_shape:
            setattr(self.app, attr_target, arr)
            return

        # Coerce by copy into target (pad/truncate per-dimension)
        out = np.full(target_shape, np.nan, dtype=dtype)
        try:
            src_slices, dst_slices = [], []
            for i, tsize in enumerate(target_shape):
                ssize = arr.shape[i] if i < arr.ndim else 1
                n = min(tsize, ssize)
                src_slices.append(slice(0, n))
                dst_slices.append(slice(0, n))
            out[tuple(dst_slices)] = arr[tuple(src_slices)]
        except Exception:
            # last-resort: flatten-copy
            n = min(out.size, arr.size)
            out.reshape(-1)[:n] = arr.reshape(-1)[:n]
        setattr(self.app, attr_target, out)

    def _enforce_all(self, T: int) -> None:
        """Create/Coerce all fields to their exact shapes."""
        for name, shp in self._target_shapes(T).items():
            self._ensure_exact(name, shp)

    # ------------------------------ Safe getters ----------------------------- #
    def _get_arr(self, attr: str) -> Optional[np.ndarray]:
        """Read array from `self.app`, resolving Qt collisions (attr -> attr_arr)."""
        attr_target = self._resolve_attr_target(attr)
        v = getattr(self.app, attr_target, None)
        if v is None or callable(v):
            return None
        try:
            return np.asarray(v)
        except Exception:
            return None

    # ------------------------------ Save: NPZ -------------------------------- #
    def _save_npz(self) -> None:
        out_npz = self.save_dir / "faceit.npz"
        d = {
            "pupil_center": self._get_arr("pupil_center"),
            "pupil_center_X": self._get_arr("pupil_center_X"),
            "pupil_center_y": self._get_arr("pupil_center_y"),
            "pupil_dilation_blinking_corrected": self._get_arr("final_pupil_area"),
            "pupil_dilation": self._get_arr("pupil_dilation"),
            "X_saccade": self._get_arr("X_saccade_updated"),
            "Y_saccade": self._get_arr("Y_saccade_updated"),
            "pupil_distance_from_corner": self._get_arr("pupil_distance_from_corner"),
            "width": self._get_arr("width"),
            "height": self._get_arr("height"),
            "motion_energy": self._get_arr("motion_energy"),
            "motion_energy_without_grooming": self._get_arr("facemotion_without_grooming"),
            "grooming_ids": self._get_arr("grooming_ids"),
            "grooming_threshold": self._get_arr("grooming_thr"),
            "blinking_ids": self._get_arr("blinking_ids"),
            "angle": self._get_arr("angle"),
            "Face_frame": self._get_arr("Face_frame"),
            "Pupil_frame": self._get_arr("Pupil_frame"),
        }
        # Replace None with empty arrays to keep savez happy
        d = {k: (v if v is not None else np.array([])) for k, v in d.items()}
        np.savez_compressed(out_npz, **d)
        logger.info("Data successfully saved to %s", out_npz)

    # ------------------------------ Save: NWB -------------------------------- #
    def _save_nwb(self, output_path: Path) -> None:
        if not self._call_bool_method(("nwb_check",)):
            logger.info("NWB check returned False; skipping NWB save.")
            return

        nwbfile = NWBFile(
            session_description="faceit",
            identifier=f"pupil_facemotion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            session_start_time=datetime.now(),
            file_create_date=datetime.now(),
        )

        proc = ProcessingModule(
            name="eye_facial_movement",
            description=("Contains pupil size, dilation, position and saccade data, "
                         "and whisker pad motion energy")
        )
        nwbfile.add_processing_module(proc)

        def add_ts(name: str, data: Optional[np.ndarray], unit: str = "a.u."):
            if data is None:
                return
            arr = np.asarray(data)
            if arr.size == 0:
                return
            arr = np.ascontiguousarray(arr)
            # time assumed on last axis (we enforce (2, T) for saccades)
            T = arr.shape[-1] if arr.ndim > 1 else len(arr)
            ts = TimeSeries(name=name, data=arr, unit=unit, timestamps=np.arange(T))
            proc.add_data_interface(ts)

        add_ts("pupil_center", self._get_arr("pupil_center"))
        add_ts("pupil_center_X", self._get_arr("pupil_center_X"))
        add_ts("pupil_center_y", self._get_arr("pupil_center_y"))
        add_ts("pupil_dilation", self._get_arr("pupil_dilation"))
        add_ts("pupil_dilation_blinking_corrected", self._get_arr("final_pupil_area"))
        add_ts("X_saccade", self._get_arr("X_saccade_updated"))
        add_ts("Y_saccade", self._get_arr("Y_saccade_updated"))
        add_ts("pupil_distance_from_corner", self._get_arr("pupil_distance_from_corner"))
        add_ts("width", self._get_arr("width"))
        add_ts("height", self._get_arr("height"))
        add_ts("motion_energy", self._get_arr("motion_energy"))
        add_ts("motion_energy_without_grooming", self._get_arr("facemotion_without_grooming"))
        add_ts("grooming_ids", self._get_arr("grooming_ids"), unit="frame")
        add_ts("grooming_threshold", self._get_arr("grooming_thr"), unit="a.u.")
        add_ts("blinking_ids", self._get_arr("blinking_ids"), unit="frame")
        add_ts("pupil_angle", self._get_arr("angle"), unit="deg")
        add_ts("Face_frame", self._get_arr("Face_frame"), unit="frame")
        add_ts("Pupil_frame", self._get_arr("Pupil_frame"), unit="frame")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with NWBHDF5IO(str(output_path), "w") as io:
            io.write(nwbfile)
        logger.info("NWB saved to %s", output_path)

    # ---------------------------- Save: Figures ------------------------------- #
    def _save_figures(self) -> None:
        self._maybe_plot_line(
            data=self._get_arr("final_pupil_area"),
            label="blinking_corrected",
            color="firebrick",
            filename="blinking_corrected.png",
            saccade=self._get_arr("X_saccade_updated"),
        )
        self._maybe_plot_line(
            data=self._get_arr("pupil_dilation"),
            label="pupil_dilation",
            color="olive",
            filename="pupil_area.png",
            saccade=self._get_arr("X_saccade_updated"),
        )
        self._maybe_plot_line(
            data=self._get_arr("motion_energy"),
            label="motion_energy",
            color="salmon",
            filename="motion_energy.png",
            draw_groom_thr=True,
        )
        self._maybe_plot_line(
            data=self._get_arr("facemotion_without_grooming"),
            label="facemotion_without_grooming",
            color="grey",
            filename="facemotion_without_grooming.png",
        )

        # Pupil center X/Y dual panel
        x = self._get_arr("pupil_center_X")
        y = self._get_arr("pupil_center_y")
        if x is not None and y is not None and x.size and y.size:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            t = np.arange(len(x))
            ax1.plot(t, x, color="teal", label="Pupil center X")
            ax1.set_ylabel("X (px)"); ax1.set_title("Pupil center X over time")
            ax1.grid(alpha=0.3); ax1.legend()
            ax2.plot(t, y, color="darkorange", label="Pupil center Y")
            ax2.set_ylabel("Y (px)"); ax2.set_xlabel("Frame")
            ax2.set_title("Pupil center Y over time")
            ax2.grid(alpha=0.3); ax2.legend()
            fig.tight_layout()
            p = self.save_dir / "pupil_center_xy.png"
            fig.savefig(p, dpi=300)
            plt.close(fig)
            logger.info("Saved %s", p)

    # --------------------------- Plot primitives ------------------------------ #
    def _maybe_plot_line(
        self,
        data: Optional[np.ndarray],
        label: str,
        color: str,
        filename: str,
        saccade: Optional[np.ndarray] = None,
        draw_groom_thr: bool = False,
    ) -> None:
        if data is None:
            return
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return

        fig, ax = plt.subplots(figsize=(9, 3))
        self._plot_line(ax, arr, label, color)

        if saccade is not None:
            self._overlay_saccade(ax, saccade, arr)

        if draw_groom_thr:
            thr = self._get_arr("grooming_thr")
            if thr is not None:
                thr = np.asarray(thr, dtype=float)
                thr = thr[np.isfinite(thr)]
                if thr.size:
                    ax.axhline(y=float(np.nanmean(thr)), color="black", linewidth=1.2,
                               label="Grooming threshold")
                    ax.legend()

        fig.tight_layout()
        out = self.save_dir / filename
        fig.savefig(out, dpi=300)
        plt.close(fig)
        logger.info("Saved %s", out)

    @staticmethod
    def _plot_line(ax: plt.Axes, data: np.ndarray, label: str, color: str) -> None:
        t = np.arange(len(data))
        ax.plot(t, data, color=color, label=label, linestyle="--")
        ax.set_title(label.replace("_", " ").capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Frame"); ax.set_ylabel("Value")
        ax.grid(alpha=0.3); ax.legend()

    @staticmethod
    def _overlay_saccade(ax: plt.Axes, saccade: np.ndarray, main: np.ndarray) -> None:
        try:
            sac = np.asarray(saccade, dtype=float)
            if sac.ndim == 1:
                if sac.shape[0] != len(main):
                    return
                sac = sac[None, :]  # (1, T)
            elif sac.ndim == 2:
                # legacy first-column trimming
                if sac.shape[1] == len(main) + 1:
                    sac = sac[:, 1:]
                if sac.shape[1] != len(main):
                    return
            else:
                return

            data_max = np.nanmax(main); data_min = np.nanmin(main)
            rng = max(1e-9, data_max - data_min)
            y_min = data_max + 0.10 * rng; y_max = data_max + 0.20 * rng
            t = np.arange(len(main))
            ax.pcolormesh(t, [y_min, y_max], sac, cmap="RdYlGn", shading="auto")
        except Exception:
            logger.debug("Skipping saccade overlay due to an error.", exc_info=True)

    # ------------------------------ Utilities -------------------------------- #
    def _call_bool_method(self, candidates: Tuple[str, ...]) -> bool:
        for name in candidates:
            fn = getattr(self.app, name, None)
            if callable(fn):
                try:
                    return bool(fn())
                except Exception:
                    logger.warning("Method %s() raised; treating as False.", name, exc_info=True)
                    return False
        return False
