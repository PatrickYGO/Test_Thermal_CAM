"""Thermal camera viewer for the PureThermal 3 - FLIR Lepton module.

This application opens a connection to the camera, visualises the thermal
stream with a colour scale, and provides UI controls for capturing photos and
videos. It also overlays statistics such as min/max temperature and the
temperature under the mouse pointer.

The program requires a PureThermal 3 board with a Lepton sensor connected to
the computer. For development without the hardware the program will simply
display an error message.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

try:
    import uvc  # type: ignore
except Exception as exc:  # pragma: no cover - module not available in CI
    uvc = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cv2


FRAME_WIDTH = 160
FRAME_HEIGHT = 120
DEFAULT_FPS_OPTIONS = [9, 27]


class ThermalCameraError(RuntimeError):
    """Raised when the thermal camera cannot be accessed."""


@dataclass
class ThermalFrame:
    """Container for a single thermal frame in degrees Celsius."""

    data: np.ndarray
    timestamp: float


class ThermalCamera:
    """Encapsulates the communication with the PureThermal 3 board."""

    def __init__(self) -> None:
        if uvc is None:  # pragma: no cover - hardware specific
            raise ThermalCameraError(
                "The 'pylibuvc' package is not available. Install pylibuvc first."
            ) from _IMPORT_ERROR

        self._ctx: Optional[uvc.Context] = None
        self._device: Optional[uvc.DeviceHandle] = None
        self._stream_ctrl: Optional[uvc.StreamCtrl] = None
        self._lock = threading.Lock()

    def open(self, fps: int = 9) -> None:
        """Initialise the libuvc context and start streaming frames."""

        try:
            self._ctx = uvc.Context()
        except Exception as exc:  # pragma: no cover - hardware specific
            raise ThermalCameraError("Unable to initialise UVC context") from exc

        device_handle = self._find_device()
        if device_handle is None:
            self.close()
            raise ThermalCameraError(
                "PureThermal 3 camera not found. Ensure it is connected and "
                "that you have permission to access USB devices."
            )

        self._device = device_handle

        try:
            self._stream_ctrl = self._device.get_stream_ctrl_format_size(
                fourcc="Y16 ",
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                fps=fps,
            )
            self._device.start_streaming(self._stream_ctrl)
        except Exception as exc:  # pragma: no cover - hardware specific
            self.close()
            raise ThermalCameraError("Unable to start streaming from the camera") from exc

    def _find_device(self) -> Optional["uvc.DeviceHandle"]:  # pragma: no cover - hardware specific
        assert self._ctx is not None
        # Try using find_device if available.
        for vendor_key in ("vendor", "vendor_id", "vid"):
            for product_key in ("product", "product_id", "pid"):
                try:
                    device = self._ctx.find_device(**{vendor_key: 0x1E4E, product_key: 0x0100})
                except TypeError:
                    # Some versions take positional arguments (vid, pid)
                    try:
                        device = self._ctx.find_device(0x1E4E, 0x0100)
                    except Exception:
                        device = None
                except Exception:
                    device = None
                if device is not None:
                    try:
                        return device.open()
                    except Exception:
                        continue

        # Fall back to iterating over the device list if the helper failed.
        device_list_attr = getattr(self._ctx, "device_list", None)
        devices = device_list_attr() if callable(device_list_attr) else device_list_attr
        if devices is None:
            return None
        for dev in devices:
            try:
                vid = dev.get("vid") or dev.get("vendor_id") or dev.get("vendor")
                pid = dev.get("pid") or dev.get("product_id") or dev.get("product")
            except AttributeError:
                continue
            if vid == 0x1E4E and pid == 0x0100:
                try:
                    return dev.open()
                except Exception:
                    continue
        return None

    def get_frame(self, timeout: float = 1.0) -> Optional[ThermalFrame]:
        """Retrieve the latest frame as degrees Celsius."""

        if self._device is None:
            return None

        with self._lock:
            try:
                frame = self._device.get_frame_ushort(timeout=timeout)
            except uvc.UVCError as exc:  # pragma: no cover - hardware specific
                if getattr(exc, "errno", None) == getattr(uvc.UVCError, "TIMEOUT", None):
                    return None
                raise

        if frame is None:
            return None

        data = frame.asarray().reshape(frame.height, frame.width)
        # Raw data is in centi-Kelvin. Convert to degrees Celsius.
        celsius = (data.astype(np.float32) - 27315.0) / 100.0
        return ThermalFrame(data=celsius, timestamp=time.time())

    def stop(self) -> None:
        """Stop streaming if it is currently running."""

        with self._lock:
            if self._device is not None:
                try:
                    self._device.stop_streaming()
                except Exception:  # pragma: no cover - hardware specific
                    pass

    def close(self) -> None:
        """Release all resources associated with the camera."""

        self.stop()
        if self._device is not None:
            try:
                self._device.close()
            except Exception:  # pragma: no cover - hardware specific
                pass
        if self._ctx is not None:
            try:
                self._ctx.close()
            except Exception:  # pragma: no cover - hardware specific
                pass

        self._device = None
        self._ctx = None
        self._stream_ctrl = None


class ThermalWorker(QtCore.QThread):
    """Background worker that continually pulls frames from the camera."""

    frame_ready = QtCore.pyqtSignal(ThermalFrame)
    error = QtCore.pyqtSignal(str)

    def __init__(self, fps: int = 9, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._fps = fps
        self._camera: Optional[ThermalCamera] = None
        self._stopping = threading.Event()

    def run(self) -> None:  # pragma: no cover - requires hardware
        try:
            self._camera = ThermalCamera()
            self._camera.open(self._fps)
        except ThermalCameraError as exc:
            self.error.emit(str(exc))
            return

        while not self._stopping.is_set():
            try:
                frame = self._camera.get_frame(timeout=1.0)
            except Exception as exc:
                self.error.emit(f"Camera error: {exc}")
                break

            if frame is None:
                continue

            self.frame_ready.emit(frame)

        if self._camera is not None:
            self._camera.close()

    def stop(self) -> None:  # pragma: no cover - requires hardware
        self._stopping.set()
        self.wait(2000)


class ThermalImageDisplay(QtWidgets.QLabel):
    """Label used to display the thermal image while tracking mouse position."""

    cursorMoved = QtCore.pyqtSignal(float, float)
    cursorLeft = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setScaledContents(True)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # pragma: no cover - UI only
        if self.pixmap() is None or self.width() == 0 or self.height() == 0:
            return
        x_ratio = max(0.0, min(1.0, event.pos().x() / self.width()))
        y_ratio = max(0.0, min(1.0, event.pos().y() / self.height()))
        self.cursorMoved.emit(x_ratio, y_ratio)

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # pragma: no cover - UI only
        del event
        self.cursorLeft.emit()


class ColorbarCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas used to draw the colour scale."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._figure = Figure(figsize=(1.5, 4.0), dpi=100)
        self._axes = self._figure.add_axes([0.35, 0.05, 0.3, 0.9])
        super().__init__(self._figure)
        self.setParent(parent)
        self.update_scale(0.0, 100.0)

    def update_scale(self, min_temp: float, max_temp: float) -> None:
        if np.isclose(max_temp, min_temp):
            max_temp += 0.1
        gradient = np.linspace(max_temp, min_temp, 256).reshape(-1, 1)
        self._axes.clear()
        self._axes.imshow(gradient, aspect="auto", cmap=cm.get_cmap("inferno"))
        self._axes.set_xticks([])
        ticks = np.linspace(0, 255, 5)
        labels = np.linspace(max_temp, min_temp, 5)
        self._axes.set_yticks(ticks)
        self._axes.set_yticklabels([f"{t:.1f}°C" for t in labels])
        self.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    """Main UI that combines the viewer, controls, and overlays."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PureThermal 3 - FLIR Lepton Viewer")

        self._image_display = ThermalImageDisplay()
        self._image_display.cursorMoved.connect(self._on_cursor_moved)
        self._image_display.cursorLeft.connect(self._on_cursor_left)

        self._colorbar = ColorbarCanvas()

        self._capture_button = QtWidgets.QPushButton("Capturer une photo")
        self._capture_button.clicked.connect(self._capture_photo)

        self._record_button = QtWidgets.QPushButton("Démarrer l'enregistrement")
        self._record_button.setCheckable(True)
        self._record_button.clicked.connect(self._toggle_recording)

        self._fps_combo = QtWidgets.QComboBox()
        for fps in DEFAULT_FPS_OPTIONS:
            self._fps_combo.addItem(f"{fps} FPS", fps)
        self._fps_combo.setCurrentIndex(0)
        self._fps_combo.currentIndexChanged.connect(self._restart_camera)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(QtWidgets.QLabel("Fréquence vidéo:"))
        controls_layout.addWidget(self._fps_combo)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self._capture_button)
        controls_layout.addWidget(self._record_button)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self._image_display, stretch=5)
        main_layout.addWidget(self._colorbar, stretch=1)

        central_widget = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central_widget)
        central_layout.addLayout(main_layout)
        central_layout.addLayout(controls_layout)
        self.setCentralWidget(central_widget)

        self._worker: Optional[ThermalWorker] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._pointer_position: Optional[Tuple[int, int]] = None
        self._pointer_temperature: Optional[float] = None
        self._video_writer: Optional["cv2.VideoWriter"] = None
        self._record_fps: int = DEFAULT_FPS_OPTIONS[0]

        self._start_camera()

    # --- Camera management -------------------------------------------------

    def _start_camera(self) -> None:
        fps = int(self._fps_combo.currentData())
        self._worker = ThermalWorker(fps=fps)
        self._worker.frame_ready.connect(self._on_frame_received)
        self._worker.error.connect(self._handle_error)
        self._worker.start()

    def _restart_camera(self) -> None:
        self._record_fps = int(self._fps_combo.currentData())
        if self._worker is not None:
            self._worker.error.disconnect()
            self._worker.frame_ready.disconnect()
            self._worker.stop()
            self._worker = None
        self._start_camera()

    # --- Frame handling ----------------------------------------------------

    def _on_frame_received(self, frame: ThermalFrame) -> None:  # pragma: no cover - UI
        import cv2

        self._latest_frame = frame.data

        min_temp = float(np.min(frame.data))
        max_temp = float(np.max(frame.data))
        self._colorbar.update_scale(min_temp, max_temp)

        norm = np.clip((frame.data - min_temp) / max(1e-6, max_temp - min_temp), 0, 1)
        display = (norm * 255).astype(np.uint8)
        color_image = cv2.applyColorMap(display, cv2.COLORMAP_INFERNO)

        pointer_text = "--"
        if self._pointer_position is not None and self._latest_frame is not None:
            x, y = self._pointer_position
            x = np.clip(x, 0, self._latest_frame.shape[1] - 1)
            y = np.clip(y, 0, self._latest_frame.shape[0] - 1)
            self._pointer_temperature = float(self._latest_frame[y, x])
            pointer_text = f"{self._pointer_temperature:.1f}°C"
            cv2.drawMarker(color_image, (x, y), (0, 255, 255), markerType=cv2.MARKER_CROSS)

        overlay = color_image.copy()
        text_color = (255, 255, 255)
        bottom = color_image.shape[0] - 10
        cv2.putText(
            overlay,
            f"Min: {min_temp:.1f}°C",
            (10, color_image.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
        )
        cv2.putText(
            overlay,
            f"Max: {max_temp:.1f}°C",
            (10, color_image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
        )
        pointer_label = f"Pointeur: {pointer_text}"
        (text_width, text_height), _ = cv2.getTextSize(
            pointer_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.putText(
            overlay,
            pointer_label,
            (max(10, color_image.shape[1] - text_width - 10), bottom),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
        )

        if self._video_writer is not None:
            self._video_writer.write(overlay)

        rgb_image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_image.shape
        bytes_per_line = 3 * width
        qt_image = QtGui.QImage(
            rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        self._image_display.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    # --- Mouse interaction -------------------------------------------------

    def _on_cursor_moved(self, x_ratio: float, y_ratio: float) -> None:
        if self._latest_frame is None:
            return
        height, width = self._latest_frame.shape
        x = int(x_ratio * (width - 1))
        y = int(y_ratio * (height - 1))
        self._pointer_position = (x, y)

    def _on_cursor_left(self) -> None:
        self._pointer_position = None
        self._pointer_temperature = None

    # --- Capture and recording ---------------------------------------------

    def _capture_photo(self) -> None:  # pragma: no cover - UI
        if self._latest_frame is None:
            QtWidgets.QMessageBox.warning(self, "Capture", "Aucune image disponible")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Enregistrer la photo",
            self._default_filename("capture", ".png"),
            "Images (*.png *.jpg *.bmp)",
        )
        if not filename:
            return

        import cv2

        min_temp = float(np.min(self._latest_frame))
        max_temp = float(np.max(self._latest_frame))
        norm = np.clip(
            (self._latest_frame - min_temp) / max(1e-6, max_temp - min_temp), 0, 1
        )
        display = (norm * 255).astype(np.uint8)
        color_image = cv2.applyColorMap(display, cv2.COLORMAP_INFERNO)
        cv2.imwrite(filename, color_image)

    def _toggle_recording(self, checked: bool) -> None:  # pragma: no cover - UI
        if checked:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:  # pragma: no cover - UI
        if self._latest_frame is None:
            QtWidgets.QMessageBox.warning(self, "Enregistrement", "Aucune image disponible")
            self._record_button.setChecked(False)
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Enregistrer la vidéo",
            self._default_filename("video", ".avi"),
            "Vidéos (*.avi *.mp4)",
        )
        if not filename:
            self._record_button.setChecked(False)
            return

        import cv2

        self._record_fps = int(self._fps_combo.currentData())
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        height, width = self._latest_frame.shape
        self._video_writer = cv2.VideoWriter(filename, fourcc, self._record_fps, (width, height))
        if not self._video_writer.isOpened():
            self._video_writer = None
            QtWidgets.QMessageBox.critical(
                self,
                "Enregistrement",
                "Impossible d'ouvrir le fichier vidéo pour l'écriture.",
            )
            self._record_button.setChecked(False)
            return

        self._record_button.setText("Arrêter l'enregistrement")

    def _stop_recording(self) -> None:  # pragma: no cover - UI
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._record_button.setText("Démarrer l'enregistrement")
        self._record_button.setChecked(False)

    # --- Misc helpers ------------------------------------------------------

    def _default_filename(self, prefix: str, extension: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(Path.home() / f"{prefix}_{timestamp}{extension}")

    def _handle_error(self, message: str) -> None:  # pragma: no cover - UI
        QtWidgets.QMessageBox.critical(self, "Erreur caméra", message)
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - UI
        if self._worker is not None:
            self._worker.stop()
        self._stop_recording()
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
