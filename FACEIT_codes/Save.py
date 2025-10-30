from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from pynwb.base import TimeSeries

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


from pathlib import Path
from datetime import datetime

class SaveHandler:
    """
    Handles saving Face-It results to compressed NPZ, NWB, and summary figures.

    Expected app_instance attributes:
      - save_path: str | Path (set by LoadData)
      - folder_path: str | Path (NPY image dir)  [optional]
      - video_path: str | Path (full path to video) [optional]
      - video: bool, NPY: bool
      - nwb_check(), pupil_check(), face_check()
      - data arrays...
    """

    # ---- Public API ---------------------------------------------------------
    def __init__(self, app_instance, base_path: Path | None = None):
        self.app = app_instance
        # DO NOT compute save_dir here; store override and resolve lazily later
        self._override: Path | None = Path(base_path) if base_path is not None else None
        self._run_id: str | None = None

    @property
    def save_dir(self) -> Path:
        """
        Resolve the base directory *now* (after loader filled app.save_path/app.folder_path/app.video_path)
        and ensure a '<base>/faceIt' subfolder exists. Always called at save time.
        """
        base = self._resolve_base_dir()
        return self._ensure_faceit_dir(base)

    # ---- Helpers ------------------------------------------------------------
    def _resolve_base_dir(self) -> Path:
        """Pick the source dir in priority: override -> app.save_path -> folder_path -> parent(video_path) -> ~"""
        if self._override is not None:
            return self._override

        # 1) app.save_path (set in LoadData.open_image_folder / load_video)
        sp = getattr(self.app, "save_path", None)
        if isinstance(sp, (str, Path)) and str(sp).strip():
            p = Path(sp)
            if p.exists() and p.is_dir():
                return p

        # 2) NPY folder (images directory)
        fp = getattr(self.app, "folder_path", None)
        if isinstance(fp, (str, Path)) and str(fp).strip():
            p = Path(fp)
            if p.exists() and p.is_dir():
                return p

        # 3) Parent of video file (if any)
        vp = getattr(self.app, "video_path", None)
        if isinstance(vp, (str, Path)) and str(vp).strip():
            vpp = Path(vp)
            base = vpp.parent if vpp.suffix else vpp
            if base.exists():
                return base

        # 4) Fallback to home dir
        return Path.home()

    @staticmethod
    def _ensure_faceit_dir(base_path: Path) -> Path:
        """Create and return '<base>/faceIt' (lowercase as requested)."""
        out = Path(base_path) / "faceIt"
        out.mkdir(parents=True, exist_ok=True)
        return out

    # call this at the very start of init_save_data:
    def _stamp_run(self) -> None:
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


    def init_save_data(self) -> None:
        """
        Prepare data (fill missing arrays with NaNs where needed) and save:
        - compressed NPZ
        - NWB
        - quick-look PNG figures
        """
        self._stamp_run()  # sets self._run_id at save time
        _ = self.save_dir  # forces resolving base and creating '<base>/faceIt'


        # Decide master length from pupil_center if available, otherwise motion_energy
        len_data = self._infer_length()

        # ---- Ensure pupil arrays exist (if pupil_check() is False)
        if not self._call_bool_method(("pupil_check",)):
            self._ensure_series(len_data, "pupil_center")
            self._ensure_series(len_data, "pupil_center_X")
            self._ensure_series(len_data, "pupil_center_y")  # original name kept
            self._ensure_series(len_data, "final_pupil_area")
            self._ensure_series(len_data, "pupil_dilation")
            self._ensure_series(len_data, "X_saccade_updated", shape=(2, len_data))  # X/Y rows typical
            self._ensure_series(len_data, "Y_saccade_updated", shape=(2, len_data))
            self._ensure_series(len_data, "pupil_distance_from_corner")
            self._ensure_series(len_data, "width")
            self._ensure_series(len_data, "height")
            self._ensure_series(len_data, "angle")

        # ---- Ensure face arrays exist (if face_check() is False)
        if not self._call_bool_method(("face_check",)):
            self._ensure_series(len_data, "motion_energy")
            self._ensure_series(len_data, "facemotion_without_grooming")
            self._ensure_series(len_data, "grooming_ids")
            self._ensure_series(1, "grooming_thr")           # scalar threshold commonly 1-length
            self._ensure_series(1, "Face_frame")             # frame reference often 1-length

        # ---- Post conditions / defaults
        if getattr(self.app, "facemotion_without_grooming", None) is None:
            self.app.facemotion_without_grooming = getattr(self.app, "motion_energy", np.full((len_data,), np.nan))
        if not hasattr(self.app, "grooming_ids") or self.app.grooming_ids is None:
            self._ensure_series(len_data, "grooming_ids")
        if not hasattr(self.app, "grooming_thr") or self.app.grooming_thr is None:
            self._ensure_series(1, "grooming_thr")
        if not hasattr(self.app, "blinking_ids") or self.app.blinking_ids is None:
            self._ensure_series(len_data, "blinking_ids")
        if not hasattr(self.app, "Pupil_frame") or self.app.Pupil_frame is None:
            self._ensure_series(1, "Pupil_frame")

        try:
            self._save_npz()
        except Exception as e:
            logger.error("Failed to save npz: %s", e, exc_info=True)

        # ---- Save NWB
        try:
            nwb_out = self.save_dir / "faceit.nwb"
            self._save_nwb(nwb_out)
        except Exception as e:
            logger.error("Failed to save NWB: %s", e, exc_info=True)

        # ---- Save figures
        try:
            self._save_figures()
        except Exception as e:
            logger.error("Failed to save figures: %s", e, exc_info=True)

    # ---- Internal helpers ---------------------------------------------------

    def _infer_length(self) -> int:
        """Choose a consistent primary length for 1D signals."""
        if hasattr(self.app, "pupil_center") and getattr(self.app, "pupil_center") is not None:
            return int(len(self.app.pupil_center))
        if hasattr(self.app, "motion_energy") and getattr(self.app, "motion_energy") is not None:
            return int(len(self.app.motion_energy))
        # Last resort: avoid zero-length issues
        logger.warning("Could not infer data length; defaulting to 1.")
        return 1

    def _call_bool_method(self, candidates: Tuple[str, ...]) -> bool:
        """Call the first available boolean method name in `candidates` on app."""
        for name in candidates:
            fn = getattr(self.app, name, None)
            if callable(fn):
                try:
                    return bool(fn())
                except Exception:
                    logger.warning("Method %s() raised; treating as False.", name, exc_info=True)
                    return False
        return False

    def _ensure_series(self, length: int, attr: str, shape: Optional[Tuple[int, ...]] = None) -> None:
        """
        Ensure `self.app.attr` exists as a NumPy array of NaNs with given shape/length.
        If shape is provided, use it; else create 1D (length,).
        """
        if getattr(self.app, attr, None) is not None:
            return
        if shape is None:
            arr = np.full((length,), np.nan, dtype=float)
        else:
            arr = np.full(shape, np.nan, dtype=float)
        setattr(self.app, attr, arr)

    # -------------------------- Saving: NPZ ----------------------------------

    def _save_npz(self) -> None:
        """
        Saves all relevant arrays to NPZ. Optionally embeds video bytes
        """
        out_npz = self.save_dir / "faceit.npz"

        data_dict: Dict[str, np.ndarray] = dict(
            pupil_center=np.asarray(getattr(self.app, "pupil_center", np.array([]))),
            pupil_center_X=np.asarray(getattr(self.app, "pupil_center_X", np.array([]))),
            pupil_center_y=np.asarray(getattr(self.app, "pupil_center_y", np.array([]))),
            pupil_dilation_blinking_corrected=np.asarray(getattr(self.app, "final_pupil_area", np.array([]))),
            pupil_dilation=np.asarray(getattr(self.app, "pupil_dilation", np.array([]))),
            X_saccade=np.asarray(getattr(self.app, "X_saccade_updated", np.empty((0, 0)))),
            Y_saccade=np.asarray(getattr(self.app, "Y_saccade_updated", np.empty((0, 0)))),
            pupil_distance_from_corner=np.asarray(getattr(self.app, "pupil_distance_from_corner", np.array([]))),
            width=np.asarray(getattr(self.app, "width", np.array([]))),
            height=np.asarray(getattr(self.app, "height", np.array([]))),
            motion_energy=np.asarray(getattr(self.app, "motion_energy", np.array([]))),
            motion_energy_without_grooming=np.asarray(getattr(self.app, "facemotion_without_grooming", np.array([]))),
            grooming_ids=np.asarray(getattr(self.app, "grooming_ids", np.array([]))),
            grooming_threshold=np.asarray(getattr(self.app, "grooming_thr", np.array([]))),
            blinking_ids=np.asarray(getattr(self.app, "blinking_ids", np.array([]))),
            angle=np.asarray(getattr(self.app, "angle", np.array([]))),
            Face_frame=np.asarray(getattr(self.app, "Face_frame", np.array([]))),
            Pupil_frame=np.asarray(getattr(self.app, "Pupil_frame", np.array([]))),
        )


        try:
            np.savez_compressed(out_npz, **data_dict)
            logger.info("Data successfully saved to %s", out_npz)
        except Exception as e:
            logger.error("Failed to save NPZ: %s", e, exc_info=True)

    # -------------------------- Saving: NWB ----------------------------------

    def _save_nwb(self, output_path: Path) -> None:
        """
        Saves to NWB if `app.nwb_check()` returns True.
        Each TimeSeries uses timestamps matching its own length to avoid mismatches.
        """
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

        # Helper to add a TimeSeries safely
        def add_ts(name: str, data: Optional[np.ndarray], unit: str = "a.u."):
            if data is None:
                return
            arr = np.asarray(data)
            if arr.size == 0:
                return
            # Make row-major for 2D; most are 1D series
            arr = np.ascontiguousarray(arr)
            t = np.arange(arr.shape[-1]) if arr.ndim > 1 else np.arange(len(arr))
            ts = TimeSeries(name=name, data=arr, unit=unit, timestamps=t)
            proc.add_data_interface(ts)

        # Add your signals (names kept close to NPZ keys)
        add_ts("pupil_center", getattr(self.app, "pupil_center", None))
        add_ts("pupil_center_X", getattr(self.app, "pupil_center_X", None))
        add_ts("pupil_center_y", getattr(self.app, "pupil_center_y", None))
        add_ts("pupil_dilation", getattr(self.app, "pupil_dilation", None))
        add_ts("pupil_dilation_blinking_corrected", getattr(self.app, "final_pupil_area", None))
        add_ts("X_saccade", getattr(self.app, "X_saccade_updated", None))
        add_ts("Y_saccade", getattr(self.app, "Y_saccade_updated", None))
        add_ts("pupil_distance_from_corner", getattr(self.app, "pupil_distance_from_corner", None))
        add_ts("width", getattr(self.app, "width", None))
        add_ts("height", getattr(self.app, "height", None))
        add_ts("motion_energy", getattr(self.app, "motion_energy", None))
        add_ts("motion_energy_without_grooming", getattr(self.app, "facemotion_without_grooming", None))
        add_ts("grooming_ids", getattr(self.app, "grooming_ids", None), unit="frame")
        add_ts("grooming_threshold", getattr(self.app, "grooming_thr", None), unit="a.u.")
        add_ts("blinking_ids", getattr(self.app, "blinking_ids", None), unit="frame")
        add_ts("pupil_angle", getattr(self.app, "angle", None), unit="deg")  # if angle in degrees
        add_ts("Face_frame", getattr(self.app, "Face_frame", None), unit="frame")
        add_ts("Pupil_frame", getattr(self.app, "Pupil_frame", None), unit="frame")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with NWBHDF5IO(str(output_path), "w") as io:
            io.write(nwbfile)
        logger.info("NWB saved to %s", output_path)

    # -------------------------- Saving: Figures ------------------------------

    def _save_figures(self) -> None:
        """
        Saves quick-look PNGs for:
          - blinking corrected pupil area
          - raw pupil area
          - motion energy (with grooming threshold line)
          - face motion without grooming
          - pupil center X/Y
        """
        # Pupil (blinking corrected)
        self._maybe_plot_line(
            data=getattr(self.app, "final_pupil_area", None),
            label="blinking_corrected",
            color="firebrick",
            filename="blinking_corrected.png",
            saccade=getattr(self.app, "X_saccade_updated", None),
        )

        # Pupil (raw)
        self._maybe_plot_line(
            data=getattr(self.app, "pupil_dilation", None),
            label="pupil_dilation",
            color="olive",
            filename="pupil_area.png",
            saccade=getattr(self.app, "X_saccade_updated", None),
        )

        # Motion energy
        self._maybe_plot_line(
            data=getattr(self.app, "motion_energy", None),
            label="motion_energy",
            color="salmon",
            filename="motion_energy.png",
            draw_groom_thr=True,
        )

        # Face motion without grooming
        self._maybe_plot_line(
            data=getattr(self.app, "facemotion_without_grooming", None),
            label="facemotion_without_grooming",
            color="grey",
            filename="facemotion_without_grooming.png",
        )

        # Pupil center X/Y dual panel
        x = getattr(self.app, "pupil_center_X", None)
        y = getattr(self.app, "pupil_center_y", None)
        if x is not None and y is not None:
            x_arr = np.asarray(x)
            y_arr = np.asarray(y)
            if x_arr.size and y_arr.size:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
                t = np.arange(len(x_arr))
                ax1.plot(t, x_arr, color="teal", label="Pupil center X")
                ax1.set_ylabel("X (px)")
                ax1.set_title("Pupil center X over time")
                ax1.grid(alpha=0.3)
                ax1.legend()

                t2 = np.arange(len(y_arr))
                ax2.plot(t2, y_arr, color="darkorange", label="Pupil center Y")
                ax2.set_ylabel("Y (px)")
                ax2.set_xlabel("Frame")
                ax2.set_title("Pupil center Y over time")
                ax2.grid(alpha=0.3)
                ax2.legend()

                fig.tight_layout()
                p = self.save_dir / "pupil_center_xy.png"
                fig.savefig(p, dpi=300)
                plt.close(fig)
                logger.info("Saved %s", p)

    # -- plotting primitives --------------------------------------------------

    def _maybe_plot_line(
        self,
        data: Optional[np.ndarray],
        label: str,
        color: str,
        filename: str,
        saccade: Optional[np.ndarray] = None,
        draw_groom_thr: bool = False,
    ) -> None:
        """Plot a single line to PNG if `data` is a nonempty array."""
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
            thr = np.asarray(getattr(self.app, "grooming_thr", np.array([])), dtype=float)
            thr = thr[np.isfinite(thr)]
            if thr.size:
                ax.axhline(y=float(np.nanmean(thr)), color="black", linestyle="--", linewidth=1.2,
                           label="Grooming threshold")
                ax.legend()

        fig.tight_layout()
        out = self.save_dir / filename
        fig.savefig(out, dpi=300)
        plt.close(fig)
        logger.info("Saved %s", out)

    @staticmethod
    def _plot_line(ax: plt.Axes, data: np.ndarray, label: str, color: str) -> None:
        """Simple line plot with labels."""
        t = np.arange(len(data))
        ax.plot(t, data, color=color, label=label, linestyle="--")
        ax.set_title(label.replace("_", " ").capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.legend()

    @staticmethod
    def _overlay_saccade(ax: plt.Axes, saccade: np.ndarray, main: np.ndarray) -> None:
        """
        Overlay a small-height colormap strip for saccade data above the max(main).
        Accepts:
            - 1D (len == len(main)) or
            - 2D with shape (H, W) where W == len(main).
        If shape has an extra first column (legacy), trims to match.
        """
        try:
            sac = np.asarray(saccade, dtype=float)
            if sac.ndim == 1:
                if sac.shape[0] != len(main):
                    return
                sac = sac[None, :]  # make (1, T)
            elif sac.ndim == 2:
                # Legacy: if first column is an index/time column, drop it
                if sac.shape[1] == len(main) + 1:
                    sac = sac[:, 1:]
                if sac.shape[1] != len(main):
                    return
            else:
                return

            data_max = np.nanmax(main)
            data_min = np.nanmin(main)
            rng = max(1e-9, data_max - data_min)
            y_min = data_max + 0.10 * rng
            y_max = data_max + 0.20 * rng

            t = np.arange(len(main))
            ax.pcolormesh(t, [y_min, y_max], sac, cmap="RdYlGn", shading="auto")
        except Exception:
            # Never crash the figure for overlay issues
            logger.debug("Skipping saccade overlay due to an error.", exc_info=True)
