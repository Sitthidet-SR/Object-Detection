"""Stationery Detector: real-time webcam detection with PyQt6 + YOLO + OpenCV.

Run:  python app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# opencv-python on Linux points Qt at its own bundled plugins, which breaks PyQt6.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PyQt6.QtCore import QPropertyAnimation, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from detector import (
    CLASS_COLORS,
    CLASS_NAMES_TH,
    CLASSES,
    DEFAULT_WEIGHTS,
    count_by_class,
    draw_detections,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.join(BASE_DIR, "captures")
MAX_DET_ROWS = 8


def hex_color(cls: str) -> str:
    b, g, r = CLASS_COLORS[cls]
    return f"#{r:02x}{g:02x}{b:02x}"


STYLE = """
* { font-family: "Noto Sans Thai", "Sarabun", "Segoe UI", "Ubuntu", sans-serif; color: #e6e9f2; }
QMainWindow, #root { background: #0f1220; }
QToolTip { background: #1d2236; border: 1px solid #2c3352; padding: 6px; }

#appTitle { font-size: 22px; font-weight: 700; }
#appSub { font-size: 12px; color: #8a93b2; }
#logo { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #6c5ce7, stop:1 #00c2ff);
        border-radius: 12px; font-size: 22px; }

#pill { background: #1a1f33; border: 1px solid #2a3150; border-radius: 13px; padding: 4px 12px;
        font-size: 12px; color: #aab3d1; }
#pill[state="live"] { background: rgba(46, 204, 113, 0.15); border-color: #2ecc71; color: #2ecc71; }
#pill[state="loading"] { background: rgba(255, 184, 0, 0.13); border-color: #ffb800; color: #ffb800; }
#pill[state="warn"] { background: rgba(255, 107, 107, 0.13); border-color: #ff6b6b; color: #ff6b6b; }

#card { background: #171b2e; border: 1px solid #232a45; border-radius: 16px; }
#cardTitle { font-size: 13px; font-weight: 600; color: #8a93b2; letter-spacing: 0.5px; }

#video { background: #0a0c16; border-radius: 12px; color: #59607d; font-size: 16px; }

QPushButton { background: #222842; border: 1px solid #2e3657; border-radius: 10px;
              padding: 10px 16px; font-size: 14px; font-weight: 600; }
QPushButton:hover { background: #2a3152; }
QPushButton:pressed { background: #1c2138; }
QPushButton:disabled { color: #555c78; background: #1a1e31; border-color: #222740; }
QPushButton#primary { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #6c5ce7, stop:1 #4f8cff);
                      border: none; color: white; }
QPushButton#primary:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7d6ff0, stop:1 #649bff); }
QPushButton#primary[running="true"] { background: #ff5c7a; }
QPushButton#accent { background: #00b894; border: none; color: #04140f; }
QPushButton#accent:hover { background: #1fd1aa; }
QPushButton#accent:disabled { background: #1a1e31; color: #555c78; }
QPushButton#toggle:checked { background: rgba(108, 92, 231, 0.25); border-color: #6c5ce7; color: #c9c2ff; }
QPushButton#ghost { background: transparent; border: 1px dashed #343c60; color: #aab3d1; font-weight: 500; }
QPushButton#ghost:hover { border-color: #6c5ce7; color: #ffffff; }

QComboBox { background: #222842; border: 1px solid #2e3657; border-radius: 8px; padding: 6px 10px; }
QComboBox QAbstractItemView { background: #1d2236; selection-background-color: #6c5ce7; }

QSlider::groove:horizontal { height: 6px; background: #2a3150; border-radius: 3px; }
QSlider::sub-page:horizontal { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #6c5ce7, stop:1 #00c2ff);
                               border-radius: 3px; }
QSlider::handle:horizontal { background: white; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }

#countCard { background: #1d2238; border-radius: 12px; }
#countNum { font-size: 28px; font-weight: 800; }
#countName { font-size: 13px; font-weight: 600; }
#countEn { font-size: 11px; color: #7f88a8; }
#totalNum { font-size: 34px; font-weight: 800; color: white; }
#totalCard { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #3b2f9e, stop:1 #1c5fa8); border-radius: 12px; }
#muted { color: #7f88a8; font-size: 12px; }
#value { font-size: 13px; font-weight: 700; }

QProgressBar { background: #232945; border: none; border-radius: 4px; }

#toast { background: rgba(0, 184, 148, 0.92); color: #04140f; border-radius: 10px; padding: 10px 18px;
         font-weight: 700; font-size: 14px; }

QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 8px; }
QScrollBar::handle:vertical { background: #2e3657; border-radius: 4px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QStatusBar { background: #0b0e19; color: #7f88a8; }
"""


class DetectionWorker(QThread):
    """Reads the camera, runs YOLO and emits annotated frames."""

    frame_ready = pyqtSignal(object, object, float)  # annotated BGR frame, detections, fps
    model_loaded = pyqtSignal(str, bool, list)  # weights, using_fallback, supported classes
    error = pyqtSignal(str)

    def __init__(self, camera_index: int, weights: str, conf: float):
        super().__init__()
        self.camera_index = camera_index
        self.weights = weights
        self.conf = conf
        self.detect_enabled = True
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            from detector import StationeryDetector

            detector = StationeryDetector(self.weights)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"โหลดโมเดลไม่สำเร็จ: {e}")
            return
        self.model_loaded.emit(detector.weights, detector.using_fallback, detector.supported_classes)

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit(f"เปิดกล้องหมายเลข {self.camera_index} ไม่ได้")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        prev = time.time()
        fps = 0.0
        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.error.emit("อ่านภาพจากกล้องไม่ได้")
                break

            detections = detector.detect(frame, conf=self.conf) if self.detect_enabled else []
            annotated = draw_detections(frame, detections)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            self.frame_ready.emit(annotated, detections, fps)

        cap.release()


# ---------------------------------------------------------------------- widgets
def card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 14, 16, 16)
    lay.setSpacing(10)
    if title:
        t = QLabel(title)
        t.setObjectName("cardTitle")
        lay.addWidget(t)
    return frame, lay


class CountCard(QFrame):
    def __init__(self, cls: str):
        super().__init__()
        self.setObjectName("countCard")
        color = hex_color(cls)
        self.setStyleSheet(f"#countCard {{ border-left: 4px solid {color}; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        names = QVBoxLayout()
        names.setSpacing(0)
        th = QLabel(CLASS_NAMES_TH[cls])
        th.setObjectName("countName")
        en = QLabel(cls)
        en.setObjectName("countEn")
        names.addWidget(th)
        names.addWidget(en)
        self.num = QLabel("0")
        self.num.setObjectName("countNum")
        self.num.setStyleSheet(f"color: {color};")
        self.num.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addLayout(names, 1)
        lay.addWidget(self.num)

    def set_count(self, n: int):
        self.num.setText(str(n))


class DetRow(QWidget):
    """One detection: class name, confidence % and a colored bar."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.name = QLabel()
        self.name.setObjectName("value")
        self.pct = QLabel()
        self.pct.setObjectName("muted")
        top.addWidget(self.name, 1)
        top.addWidget(self.pct)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        lay.addLayout(top)
        lay.addWidget(self.bar)

    def set(self, cls: str, conf: float):
        color = hex_color(cls)
        self.name.setText(f"{CLASS_NAMES_TH[cls]}  ·  {cls}")
        self.pct.setText(f"{conf * 100:.1f}%")
        self.bar.setValue(int(conf * 100))
        self.bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stationery Detector")
        self.resize(1320, 800)
        self.setMinimumSize(1100, 680)

        self.worker: DetectionWorker | None = None
        self.weights = DEFAULT_WEIGHTS
        self.last_frame: np.ndarray | None = None
        self.last_counts = {c: 0 for c in CLASSES}
        self.last_dets = []

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(16)

        outer.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._build_video_panel(), 1)
        body.addWidget(self._build_side_panel())
        outer.addLayout(body, 1)

        self.toast = QLabel(self.video)
        self.toast.setObjectName("toast")
        self.toast.hide()
        self._toast_fx = QGraphicsOpacityEffect(self.toast)
        self.toast.setGraphicsEffect(self._toast_fx)

        self.statusBar().showMessage("พร้อมใช้งาน · กด Space เพื่อบันทึกภาพ")
        self._update_model_label()
        self._set_state("idle")

    # ------------------------------------------------------------------ layout
    def _build_header(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        logo = QLabel("✏️")
        logo.setObjectName("logo")
        logo.setFixedSize(46, 46)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        t = QLabel("Stationery Detector")
        t.setObjectName("appTitle")
        s = QLabel("ตรวจจับและจำแนกเครื่องเขียนแบบเรียลไทม์ด้วย YOLO")
        s.setObjectName("appSub")
        titles.addWidget(t)
        titles.addWidget(s)

        self.pill_state = QLabel()
        self.pill_state.setObjectName("pill")
        self.pill_fps = QLabel("FPS  –")
        self.pill_fps.setObjectName("pill")
        self.pill_model = QLabel()
        self.pill_model.setObjectName("pill")

        lay.addWidget(logo)
        lay.addLayout(titles)
        lay.addStretch(1)
        lay.addWidget(self.pill_model)
        lay.addWidget(self.pill_fps)
        lay.addWidget(self.pill_state)
        return w

    def _build_video_panel(self) -> QWidget:
        frame, lay = card()
        lay.setContentsMargins(12, 12, 12, 12)

        self.video = QLabel("📷\n\nกด  “เริ่มตรวจจับ”  เพื่อเปิดกล้อง")
        self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(640, 420)
        self.video.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        lay.addWidget(self.video, 1)

        bar = QHBoxLayout()
        bar.setSpacing(10)
        self.btn_start = QPushButton("▶  เริ่มตรวจจับ")
        self.btn_start.setObjectName("primary")
        self.btn_start.setMinimumWidth(170)
        self.btn_start.clicked.connect(self.toggle_camera)

        self.btn_detect = QPushButton("◉  แสดงกรอบ")
        self.btn_detect.setObjectName("toggle")
        self.btn_detect.setCheckable(True)
        self.btn_detect.setChecked(True)
        self.btn_detect.setToolTip("เปิด/ปิดการตรวจจับ (ภาพกล้องยังแสดงอยู่)")
        self.btn_detect.toggled.connect(self.toggle_detect)

        self.btn_capture = QPushButton("📸  บันทึกภาพ")
        self.btn_capture.setObjectName("accent")
        self.btn_capture.setToolTip("บันทึกภาพผลการตรวจจับ (Space)")
        self.btn_capture.clicked.connect(self.capture)
        self.btn_capture.setEnabled(False)

        self.btn_open_dir = QPushButton("📁  ภาพที่บันทึก")
        self.btn_open_dir.clicked.connect(self.open_capture_dir)

        for b in (self.btn_start, self.btn_detect, self.btn_capture, self.btn_open_dir):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keep Space for capture
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_detect)
        bar.addStretch(1)
        bar.addWidget(self.btn_open_dir)
        bar.addWidget(self.btn_capture)
        lay.addLayout(bar)
        return frame

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("root")
        side = QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 6, 0)
        side.setSpacing(14)

        # counters
        cnt, cl = card("จำนวนวัตถุที่ตรวจพบ")
        total = QFrame()
        total.setObjectName("totalCard")
        tl = QHBoxLayout(total)
        tl.setContentsMargins(14, 10, 14, 10)
        tcol = QVBoxLayout()
        tcol.setSpacing(0)
        tt = QLabel("ทั้งหมด")
        tt.setStyleSheet("font-weight:600; color:#dfe3ff; background:transparent;")
        ts = QLabel("วัตถุในเฟรมปัจจุบัน")
        ts.setStyleSheet("font-size:11px; color:#b9c2ef; background:transparent;")
        tcol.addWidget(tt)
        tcol.addWidget(ts)
        self.lbl_total = QLabel("0")
        self.lbl_total.setObjectName("totalNum")
        self.lbl_total.setStyleSheet("background:transparent;")
        tl.addLayout(tcol, 1)
        tl.addWidget(self.lbl_total)
        cl.addWidget(total)

        grid = QGridLayout()
        grid.setSpacing(8)
        self.count_cards: dict[str, CountCard] = {}
        for i, cls in enumerate(CLASSES):
            c = CountCard(cls)
            self.count_cards[cls] = c
            if i == len(CLASSES) - 1 and len(CLASSES) % 2:
                grid.addWidget(c, i // 2, 0, 1, 2)
            else:
                grid.addWidget(c, i // 2, i % 2)
        cl.addLayout(grid)
        side.addWidget(cnt)

        # detections
        det, dl = card("ค่าความมั่นใจ (Confidence)")
        self.det_rows = [DetRow() for _ in range(MAX_DET_ROWS)]
        for r in self.det_rows:
            r.hide()
            dl.addWidget(r)
        self.lbl_nodet = QLabel("ยังไม่พบวัตถุ")
        self.lbl_nodet.setObjectName("muted")
        self.lbl_nodet.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_nodet.setMinimumHeight(40)
        dl.addWidget(self.lbl_nodet)
        side.addWidget(det)

        # settings
        st, sl = card("ตั้งค่า")
        row = QHBoxLayout()
        lbl = QLabel("กล้อง")
        lbl.setObjectName("muted")
        self.combo_cam = QComboBox()
        for i in range(6):
            self.combo_cam.addItem(f"กล้อง {i}" + ("  (ค่าเริ่มต้น)" if i == 0 else ""), i)
        row.addWidget(lbl)
        row.addWidget(self.combo_cam, 1)
        sl.addLayout(row)

        crow = QHBoxLayout()
        clbl = QLabel("ความมั่นใจขั้นต่ำ")
        clbl.setObjectName("muted")
        self.lbl_conf = QLabel("50%")
        self.lbl_conf.setObjectName("value")
        crow.addWidget(clbl, 1)
        crow.addWidget(self.lbl_conf)
        sl.addLayout(crow)
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(5, 95)
        self.slider_conf.setValue(50)
        self.slider_conf.valueChanged.connect(self.change_conf)
        sl.addWidget(self.slider_conf)

        self.btn_model = QPushButton("🧠  เลือกไฟล์โมเดล (.pt)")
        self.btn_model.setObjectName("ghost")
        self.btn_model.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_model.clicked.connect(self.choose_model)
        sl.addWidget(self.btn_model)
        self.lbl_model = QLabel()
        self.lbl_model.setObjectName("muted")
        self.lbl_model.setWordWrap(True)
        sl.addWidget(self.lbl_model)
        side.addWidget(st)
        side.addStretch(1)

        for w in (self.combo_cam, self.slider_conf, self.btn_model):
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(370)
        return scroll

    # ------------------------------------------------------------------ state
    @staticmethod
    def _repolish(w: QWidget):
        w.style().unpolish(w)
        w.style().polish(w)

    def _set_state(self, state: str):
        labels = {"idle": "●  หยุดอยู่", "loading": "●  กำลังโหลด", "live": "●  LIVE", "warn": "●  มีปัญหา"}
        self.pill_state.setText(labels[state])
        self.pill_state.setProperty("state", state)
        self._repolish(self.pill_state)

    def _set_running_ui(self, running: bool):
        self.btn_start.setText("■  หยุด" if running else "▶  เริ่มตรวจจับ")
        self.btn_start.setProperty("running", "true" if running else "false")
        self._repolish(self.btn_start)
        self.combo_cam.setEnabled(not running)
        self.btn_model.setEnabled(not running)

    # ------------------------------------------------------------------ camera
    def toggle_camera(self):
        if self.worker and self.worker.isRunning():
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self._set_state("loading")
        self.statusBar().showMessage("กำลังโหลดโมเดลและเปิดกล้อง...")
        self.worker = DetectionWorker(self.combo_cam.currentData(), self.weights, self.slider_conf.value() / 100)
        self.worker.detect_enabled = self.btn_detect.isChecked()
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.model_loaded.connect(self.on_model_loaded)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        self._set_running_ui(True)

    def stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(3000)

    def on_worker_finished(self):
        self._set_running_ui(False)
        if self.pill_state.property("state") != "warn":
            self._set_state("idle")
            self.statusBar().showMessage("ปิดกล้องแล้ว")
        self.pill_fps.setText("FPS  –")
        self.btn_capture.setEnabled(self.last_frame is not None)

    def toggle_detect(self, on: bool):
        self.btn_detect.setText("◉  แสดงกรอบ" if on else "○  ซ่อนกรอบ")
        if self.worker:
            self.worker.detect_enabled = on

    def change_conf(self, value: int):
        self.lbl_conf.setText(f"{value}%")
        if self.worker:
            self.worker.conf = value / 100

    # ------------------------------------------------------------------ slots
    def on_model_loaded(self, weights: str, fallback: bool, supported: list):
        self._set_state("live")
        if fallback:
            self.pill_model.setText("⚠  โมเดล COCO (เฉพาะกรรไกร)")
            self.statusBar().showMessage("ไม่พบ models/best.pt จึงใช้โมเดล COCO แทน (ตรวจได้เฉพาะกรรไกร)")
            QMessageBox.warning(
                self,
                "ยังไม่มีโมเดลที่เทรนเอง",
                "ไม่พบไฟล์ models/best.pt\n\n"
                "โปรแกรมจะใช้โมเดล YOLO มาตรฐาน (COCO) แทน ซึ่งรู้จักแค่ 'กรรไกร'\n"
                "หากต้องการตรวจจับครบ 5 ชนิด ให้เทรนโมเดลด้วย train.py ตามคู่มือ README_TH.md",
            )
        else:
            self.pill_model.setText(f"🧠  {os.path.basename(weights)} · {len(supported)} คลาส")
            self.statusBar().showMessage(f"โหลดโมเดลแล้ว · คลาส: {', '.join(supported)}")

    def on_frame(self, frame: np.ndarray, dets: list, fps: float):
        counts = count_by_class(dets)
        self.last_frame = frame
        self.last_dets = dets
        self.last_counts = counts
        self.btn_capture.setEnabled(True)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.video.setPixmap(pix)

        for cls, c in self.count_cards.items():
            c.set_count(counts.get(cls, 0))
        self.lbl_total.setText(str(sum(counts.values())))
        self.pill_fps.setText(f"FPS  {fps:.1f}")

        top = sorted(dets, key=lambda d: -d.conf)[:MAX_DET_ROWS]
        for i, row in enumerate(self.det_rows):
            if i < len(top):
                row.set(top[i].cls, top[i].conf)
                row.show()
            else:
                row.hide()
        self.lbl_nodet.setVisible(not top)

    def on_error(self, msg: str):
        self._set_state("warn")
        self.statusBar().showMessage(msg)
        QMessageBox.critical(self, "เกิดข้อผิดพลาด", msg)

    # ------------------------------------------------------------------ actions
    def capture(self):
        if self.last_frame is None:
            return
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(CAPTURE_DIR, f"capture_{stamp}.jpg")
        cv2.imwrite(img_path, self.last_frame)

        info = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "image": os.path.basename(img_path),
            "counts": self.last_counts,
            "detections": [
                {"class": d.cls, "confidence": round(d.conf, 4), "box_xyxy": d.box} for d in self.last_dets
            ],
        }
        with open(img_path.replace(".jpg", ".json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        self.statusBar().showMessage(f"บันทึกภาพแล้ว: {img_path}")
        self.show_toast(f"✓  บันทึกภาพแล้ว  ·  {os.path.basename(img_path)}")

    def show_toast(self, text: str):
        self.toast.setText(text)
        self.toast.adjustSize()
        self.toast.move((self.video.width() - self.toast.width()) // 2, self.video.height() - self.toast.height() - 20)
        self._toast_fx.setOpacity(1.0)
        self.toast.show()
        self.toast.raise_()
        anim = QPropertyAnimation(self._toast_fx, b"opacity", self)
        anim.setDuration(600)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.finished.connect(self.toast.hide)
        self._toast_anim = anim
        QTimer.singleShot(1400, anim.start)

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกไฟล์โมเดล YOLO", BASE_DIR, "YOLO weights (*.pt *.onnx)")
        if path:
            self.weights = path
            self._update_model_label()

    def open_capture_dir(self):
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(CAPTURE_DIR))

    def _update_model_label(self):
        exists = os.path.exists(self.weights)
        name = os.path.relpath(self.weights, BASE_DIR)
        self.lbl_model.setText(f"ไฟล์: {name}" + ("" if exists else "\n⚠ ไม่พบไฟล์ จะใช้โมเดล COCO แทน"))
        self.pill_model.setText(f"🧠  {os.path.basename(self.weights)}" if exists else "⚠  ยังไม่มีโมเดลที่เทรน")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.capture()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
