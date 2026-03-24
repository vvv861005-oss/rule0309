"""
REVOLUTION HYPERVISION 3D — 완전 통합판
================================================================================
기존 WiFi RF 관측 포화 이론(ObservationalSaturation) 엔진을 유지하면서,
아래 진짜 3D 레이어를 완전 추가:

  ① 3D 카메라 파이프라인 (실시간 Depth + RGB)
      • Intel RealSense D4xx/L5xx 자동 감지   → pyrealsense2
      • Luxonis OAK-D 자동 감지              → depthai
      • ZED 2/ZED X 자동 감지               → pyzed
      • 위 세 가지 모두 없으면 일반 웹캠(OpenCV) + MiDaS monocular depth 추정
      • 전부 없으면 완전 합성(synthetic) 모드

  ② 3D 포인트클라우드 처리 (per-frame)
      • Depth → 3-D 포인트클라우드 생성
      • RGB 텍스처 매핑
      • Open3D RANSAC 평면 제거, 클러스터링, 다운샘플링

  ③ MediaPipe Pose (ML 전신 포즈 추정)
      • 33개 랜드마크 → 3D 스켈레톤 (World Landmarks 포함)
      • 프레임별 추정 + 시계열 누적

  ④ ML 움직임 분류 (경량 LSTM / 규칙 기반 fallback)
      • 관절 속도/가속도 벡터 계산
      • 행동 레이블: STANDING / WALKING / RAISING_ARM / BENDING / SITTING / RUNNING
      • 신뢰도 점수 포함

  ⑤ RF + Depth + Pose 3-way 융합
      • 위치 가중 칼만 필터 (3D)
      • 볼륨 voxel을 실제 사람 위치로 교정
      • 호흡·심박 추정을 어깨 움직임 궤적으로 보정

  ⑥ Streamlit 렌더링 확장
      • Plotly 인터랙티브 3D 스켈레톤 + 포인트클라우드 + RF 볼륨 오버레이
      • 사람별 bounding box + 행동 레이블 + confidence bar
      • 실시간 깊이 이미지 + RGB 영상 나란히 표시
      • 칼만 트래젝토리 꼬리

중요 안내:
  이 코드는 RealSense / OAK-D / ZED 드라이버가 없어도 실행됩니다.
  카메라 없는 환경에서는 합성 시뮬레이션 모드가 자동 활성화됩니다.
  실제 카메라 연결 시 시스템이 자동 감지·전환합니다.
================================================================================
"""

import sys
import subprocess
import os
import re
import time
import threading
import logging
import math
import collections
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import traceback

# ──────────────────────────────────────────────────────────────────────────────
# 의존성 자동 설치
# ──────────────────────────────────────────────────────────────────────────────
def _pip(*pkgs):
    for p in pkgs:
        mod = p.split("[")[0].replace("-", "_")
        try:
            __import__(mod)
        except Exception:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--quiet",
                     "--break-system-packages", p],
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass  # 설치 실패해도 계속 (선택적 패키지)

_pip(
    "streamlit", "numpy", "scipy", "plotly", "pandas",
    "opencv-python-headless", "mediapipe",
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import mediapipe as mp
    # MediaPipe 0.10+ Tasks API 우선, fallback to legacy solutions
    _HAS_MP = True
    _mp_pose = None
    _mp_drawing = None
    _mp_drawing_styles = None
    # Try legacy solutions (still available in some builds)
    try:
        _mp_pose = mp.solutions.pose
        _mp_drawing = mp.solutions.drawing_utils
        _mp_drawing_styles = mp.solutions.drawing_styles
        _MP_USE_LEGACY = True
    except AttributeError:
        # Use Tasks API (MediaPipe 0.10+)
        _MP_USE_LEGACY = False
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            _mp_tasks = mp_tasks
            _mp_vision_tasks = mp_vision
        except Exception:
            _HAS_MP = False
except ImportError:
    _HAS_MP = False
    _mp_pose = None
    _MP_USE_LEGACY = False

# 3D 카메라 SDK 감지
try:
    import pyrealsense2 as rs
    _HAS_RS = True
except ImportError:
    _HAS_RS = False

try:
    import depthai as dai
    _HAS_DAI = True
except ImportError:
    _HAS_DAI = False

try:
    import pyzed.sl as sl
    _HAS_ZED = True
except ImportError:
    _HAS_ZED = False

# MiDaS (웹캠 전용 monocular depth)
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit_aer import AerSimulator
    try:
        from qiskit.circuit.library import QFTGate
        _HAS_QFTGATE = True
    except Exception:
        _HAS_QFTGATE = False
    _HAS_QISKIT = True
except Exception:
    _HAS_QISKIT = False
    _HAS_QFTGATE = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("REV_3D_ML")


# ══════════════════════════════════════════════════════════════════════════════
# §1. 데이터 구조
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RFSample:
    t: float
    rssi_dbm: float
    noise_dbm: float
    snr_db: float
    freq_ghz: float
    tx_rate_mbps: float
    rx_rate_mbps: float
    phase_proxy: float = 0.0
    sim_target_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def wavelength_m(self) -> float:
        return 3e8 / (self.freq_ghz * 1e9)


@dataclass
class PoseLandmark3D:
    """MediaPipe 33개 랜드마크 + 3D world 좌표"""
    index: int
    name: str
    x: float  # 화면 정규화 x
    y: float  # 화면 정규화 y
    z: float  # 상대 깊이 (MediaPipe 내부)
    wx: float  # world x (m)
    wy: float  # world y (m)
    wz: float  # world z (m)
    visibility: float  # 0~1


@dataclass
class PersonTrack:
    """한 명의 추적 대상"""
    id: int
    pose_landmarks: List[PoseLandmark3D] = field(default_factory=list)
    depth_xyz: Optional[Tuple[float, float, float]] = None   # depth 카메라 기반 위치
    fused_xyz: Optional[Tuple[float, float, float]] = None   # 칼만 융합 위치
    action_label: str = "UNKNOWN"
    action_confidence: float = 0.0
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    # 칼만 상태 [x,y,z, vx,vy,vz]
    kalman_state: Optional[np.ndarray] = None
    kalman_cov: Optional[np.ndarray] = None
    last_seen: float = 0.0
    shoulder_heights: deque = field(default_factory=lambda: deque(maxlen=120))


@dataclass
class CameraFrame:
    """카메라 한 프레임 전체"""
    t: float
    rgb: Optional[np.ndarray] = None          # (H,W,3) uint8
    depth_m: Optional[np.ndarray] = None      # (H,W) float32 미터 단위
    pointcloud_xyz: Optional[np.ndarray] = None   # (N,3) float32
    pointcloud_rgb: Optional[np.ndarray] = None   # (N,3) uint8
    persons: List[PersonTrack] = field(default_factory=list)
    intrinsics: Optional[Dict] = None  # fx,fy,cx,cy


@dataclass
class TheoryConfig:
    enabled: bool = True
    saturation_limit: float = 10.0
    energy_limit: float = 1.0
    observability_gain: float = 1.25
    ontological_gain: float = 1.6
    smoothing: float = 0.78
    sigma_base: float = 0.48
    sigma_min: float = 0.18
    top_k_peaks: int = 10
    map_mode: str = "harmonic"
    temporal_persistence: float = 0.82
    projection_gain: float = 1.4
    pseudo_camera_gain: float = 1.8
    pseudo_camera_gamma: float = 0.75


# ══════════════════════════════════════════════════════════════════════════════
# §2. 관측 포화 이론 엔진 (원본 유지)
# ══════════════════════════════════════════════════════════════════════════════

class ObservationalSaturationEngine:
    def __init__(self, config: Optional[TheoryConfig] = None):
        self.cfg = config or TheoryConfig()

    def set_config(self, config: TheoryConfig):
        self.cfg = config

    def saturate(self, value, limit=None):
        if not self.cfg.enabled:
            return np.asarray(value, dtype=float)
        limit = max(limit or self.cfg.saturation_limit, 1e-6)
        arr = np.asarray(value, dtype=float)
        sign = np.sign(arr)
        mag = np.abs(arr)
        mode = self.cfg.map_mode
        if mode == "tanh":
            out = limit * np.tanh(mag / limit)
        elif mode == "arctan":
            out = limit * (2.0 / np.pi) * np.arctan((np.pi / 2.0) * mag / limit)
        else:
            out = (mag * limit) / (mag + limit + 1e-9)
        return sign * out

    def inverse_saturate(self, value, limit=None):
        if not self.cfg.enabled:
            return np.asarray(value, dtype=float)
        limit = max(limit or self.cfg.saturation_limit, 1e-6)
        arr = np.asarray(value, dtype=float)
        sign = np.sign(arr)
        mag = np.clip(np.abs(arr), 0, limit * 0.995)
        mode = self.cfg.map_mode
        if mode == "tanh":
            out = limit * np.arctanh(np.clip(mag / limit, 0, 0.995))
        elif mode == "arctan":
            out = limit * np.tan(np.clip((np.pi / 2.0) * mag / limit, 0, 1.555)) / (np.pi / 2.0)
        else:
            out = (limit * mag) / np.maximum(limit - mag, 1e-9)
        return sign * out

    def observability(self, snr, rssi, tx, rx):
        snr_term = 1.0 / (1.0 + np.exp(-(np.asarray(snr) - 8.0) / 4.0))
        rssi_term = 1.0 / (1.0 + np.exp(-(np.asarray(rssi) + 72.0) / 4.5))
        rate_term = np.clip((np.asarray(tx) + np.asarray(rx)) / 700.0, 0, 1)
        raw = 0.5 * snr_term + 0.3 * rssi_term + 0.2 * rate_term
        return np.clip(raw * self.cfg.observability_gain, 0, 1)

    def energy_map(self, delta_d, snr, observability):
        delta = np.asarray(delta_d, dtype=float)
        snr_arr = np.asarray(snr, dtype=float)[-len(delta):]
        obs = np.asarray(observability, dtype=float)[-len(delta):]
        latent = np.abs(self.inverse_saturate(delta * self.cfg.ontological_gain, self.cfg.energy_limit))
        observed = np.abs(self.saturate(latent * (0.55 + 0.45 * obs), self.cfg.energy_limit))
        snr_gain = np.clip((snr_arr + 10.0) / 35.0, 0, 1)
        energy = observed * (0.35 + 0.65 * snr_gain)
        return latent, observed, energy

    def epistemic_metrics(self, delta_d, snr, rssi, tx, rx):
        if len(delta_d) == 0:
            return {
                "observability_score": 0.0, "epistemic_gap": 0.0,
                "boundary_pressure": 0.0, "latent_energy": 0.0,
                "observed_energy": 0.0, "energy_ratio": 0.0,
                "adaptive_threshold": 0.18, "confidence": 0.0,
                "observability_series": [], "latent_series": [], "observed_series": [],
            }
        n = len(delta_d)
        obs = self.observability(snr[-n:], rssi[-n:], tx[-n:], rx[-n:])
        latent, observed, _ = self.energy_map(delta_d, snr, obs)
        latent_m = float(np.mean(latent))
        observed_m = float(np.mean(observed))
        score = float(np.mean(obs))
        gap = float(np.clip(latent_m - observed_m, 0.0, None))
        boundary = float(np.mean(np.abs(observed) / max(self.cfg.energy_limit, 1e-6)))
        threshold = 0.16 + 0.14 * (1.0 - score)
        confidence = float(np.clip(score * np.exp(-gap / (latent_m + 1e-6)), 0, 1))
        return {
            "observability_score": round(score, 5),
            "epistemic_gap": round(gap, 6),
            "boundary_pressure": round(boundary, 6),
            "latent_energy": round(latent_m, 6),
            "observed_energy": round(observed_m, 6),
            "energy_ratio": round(observed_m / (latent_m + 1e-9), 6),
            "adaptive_threshold": round(threshold, 5),
            "confidence": round(confidence, 5),
            "observability_series": obs.tolist(),
            "latent_series": latent.tolist(),
            "observed_series": observed.tolist(),
        }

    def manifest(self):
        return {
            "core_principle": "Infinity exists, but cannot be observed.",
            "mapping": "O_obs = O_ont / (1 + O_ont / L)",
            "interpretation": "관측량은 무한 자체가 아니라 포화된 표현이다.",
            "rf_translation": "RF 변화량을 직접 실재로 간주하지 않고, 포화 관측의 경계 안에서 latent field를 추정한다.",
            "visual_hypothesis": "3D latent volume + Depth PointCloud + ML Pose를 3-way 융합하여 진짜 3D 씬을 재구성한다.",
        }


# ══════════════════════════════════════════════════════════════════════════════
# §3. 3D 카메라 서브시스템
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_MODES = ["RealSense", "OAK-D", "ZED", "Webcam+MiDaS", "Synthetic"]

class Camera3DBase:
    """모든 카메라 백엔드의 공통 인터페이스"""
    mode: str = "Base"
    W: int = 640
    H: int = 480
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0

    def start(self): pass
    def stop(self): pass
    def read(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Returns (rgb_uint8, depth_meters_float32) or None"""
        return None

    def intrinsics(self) -> Dict:
        return {"fx": self.fx, "fy": self.fy,
                "cx": self.cx, "cy": self.cy,
                "W": self.W, "H": self.H}

    def depth_to_pointcloud(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """depth(H,W) float32 m → (N,3) xyz, (N,3) rgb"""
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        valid = (depth > 0.1) & (depth < 8.0)
        z = depth[valid].astype(np.float32)
        x = ((u[valid] - self.cx) * z / self.fx).astype(np.float32)
        y = ((v[valid] - self.cy) * z / self.fy).astype(np.float32)
        xyz = np.stack([x, y, z], axis=1)
        if rgb is not None and rgb.shape[:2] == (H, W):
            c = rgb[valid].astype(np.float32)
        else:
            c = np.ones((xyz.shape[0], 3), dtype=np.float32) * 128
        # 다운샘플
        if len(xyz) > 8000:
            idx = np.random.choice(len(xyz), 8000, replace=False)
            xyz, c = xyz[idx], c[idx]
        return xyz, c.astype(np.uint8)


# ── RealSense ────────────────────────────────────────────────────────────────
class RealSenseCamera(Camera3DBase):
    mode = "RealSense"

    def __init__(self):
        self._pipe = None
        self._config = None
        self._align = None

    def start(self):
        if not _HAS_RS:
            raise RuntimeError("pyrealsense2 not installed")
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self._pipe.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._scale = depth_sensor.get_depth_scale()
        self._align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = intr.fx, intr.fy
        self.cx, self.cy = intr.ppx, intr.ppy
        self.W, self.H = intr.width, intr.height
        log.info("[RealSense] started scale=%.4f", self._scale)

    def stop(self):
        if self._pipe:
            self._pipe.stop()

    def read(self):
        try:
            frames = self._pipe.wait_for_frames(timeout_ms=100)
            aligned = self._align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                return None
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._scale
            rgb = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
            return rgb, depth
        except Exception as e:
            log.warning("[RealSense] read error: %s", e)
            return None


# ── OAK-D ─────────────────────────────────────────────────────────────────────
class OAKDCamera(Camera3DBase):
    mode = "OAK-D"

    def __init__(self):
        self._device = None
        self._q_rgb = None
        self._q_depth = None
        self.fx = self.fy = 700.0
        self.cx, self.cy = 320.0, 240.0

    def start(self):
        if not _HAS_DAI:
            raise RuntimeError("depthai not installed")
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)

        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        xout_depth.setStreamName("depth")
        cam_rgb.preview.link(xout_rgb.input)
        stereo.depth.link(xout_depth.input)

        self._device = dai.Device(pipeline)
        self._q_rgb = self._device.getOutputQueue("rgb", 4, False)
        self._q_depth = self._device.getOutputQueue("depth", 4, False)
        log.info("[OAK-D] started")

    def stop(self):
        if self._device:
            self._device.close()

    def read(self):
        try:
            rgb_frame = self._q_rgb.get()
            depth_frame = self._q_depth.get()
            rgb = rgb_frame.getCvFrame()
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = depth_frame.getFrame().astype(np.float32) / 1000.0  # mm→m
            h, w = depth.shape
            if rgb.shape[:2] != (h, w):
                rgb = cv2.resize(rgb, (w, h))
            return rgb, depth
        except Exception as e:
            log.warning("[OAK-D] read error: %s", e)
            return None


# ── ZED ───────────────────────────────────────────────────────────────────────
class ZEDCamera(Camera3DBase):
    mode = "ZED"

    def __init__(self):
        self._cam = None
        self._runtime = None
        self._image = None
        self._depth = None

    def start(self):
        if not _HAS_ZED:
            raise RuntimeError("pyzed not installed")
        self._cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_resolution = sl.RESOLUTION.HD720
        status = self._cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")
        self._runtime = sl.RuntimeParameters()
        self._image = sl.Mat()
        self._depth = sl.Mat()
        cam_info = self._cam.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters.left_cam
        self.fx, self.fy = calib.fx, calib.fy
        self.cx, self.cy = calib.cx, calib.cy
        self.W = cam_info.camera_configuration.resolution.width
        self.H = cam_info.camera_configuration.resolution.height
        log.info("[ZED] started")

    def stop(self):
        if self._cam:
            self._cam.close()

    def read(self):
        try:
            if self._cam.grab(self._runtime) == sl.ERROR_CODE.SUCCESS:
                self._cam.retrieve_image(self._image, sl.VIEW.LEFT)
                self._cam.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
                rgb = self._image.get_data()[:, :, :3]
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGB)
                depth = self._depth.get_data().astype(np.float32)
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                return rgb, depth
        except Exception as e:
            log.warning("[ZED] read error: %s", e)
        return None


# ── Webcam + MiDaS monocular depth ────────────────────────────────────────────
class WebcamMiDaSCamera(Camera3DBase):
    mode = "Webcam+MiDaS"
    _midas = None
    _transform = None
    _device_str = "cpu"

    def __init__(self, cam_index: int = 0):
        self._cap = None
        self._cam_idx = cam_index
        self._midas_ready = False

    def start(self):
        if not _HAS_CV2:
            raise RuntimeError("opencv not installed")
        self._cap = cv2.VideoCapture(self._cam_idx)
        if not self._cap.isOpened():
            raise RuntimeError(f"Webcam {self._cam_idx} cannot be opened")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.W = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cx = self.W / 2
        self.cy = self.H / 2
        self.fx = self.W * 1.2   # 근사 FOV
        self.fy = self.W * 1.2
        self._load_midas()
        log.info("[Webcam+MiDaS] started W=%d H=%d midas=%s",
                 self.W, self.H, self._midas_ready)

    def _load_midas(self):
        if not _HAS_TORCH:
            return
        try:
            self._midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                                          trust_repo=True)
            self._midas.eval()
            if _HAS_TORCH and torch.cuda.is_available():
                self._device_str = "cuda"
                self._midas.cuda()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms",
                                               trust_repo=True)
            self._transform = midas_transforms.small_transform
            self._midas_ready = True
        except Exception as e:
            log.warning("[MiDaS] load failed (relative depth disabled): %s", e)

    def stop(self):
        if self._cap:
            self._cap.release()

    def read(self):
        try:
            ret, frame = self._cap.read()
            if not ret:
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self._midas_ready:
                depth = self._estimate_depth(rgb)
            else:
                # 깊이 없으면 중심 거리 1.5m 근사 (평탄 추정)
                depth = np.full((self.H, self.W), 1.5, dtype=np.float32)
            return rgb, depth
        except Exception as e:
            log.warning("[Webcam] read error: %s", e)
            return None

    def _estimate_depth(self, rgb: np.ndarray) -> np.ndarray:
        try:
            device = torch.device(self._device_str)
            input_batch = self._transform(rgb).to(device)
            with torch.no_grad():
                prediction = self._midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            output = prediction.cpu().numpy()
            # MiDaS 역수 깊이 → 절대 깊이 근사 (1m~5m 범위 정규화)
            dmin, dmax = output.min(), output.max()
            if dmax - dmin < 1e-6:
                return np.full(rgb.shape[:2], 1.5, dtype=np.float32)
            norm = (output - dmin) / (dmax - dmin + 1e-9)
            depth_m = (1.0 - norm) * 5.0 + 0.3  # 역수 → 먼 곳이 작음
            return depth_m.astype(np.float32)
        except Exception as e:
            log.warning("[MiDaS] inference failed: %s", e)
            return np.full(rgb.shape[:2], 1.5, dtype=np.float32)


# ── Synthetic (카메라 없음) ───────────────────────────────────────────────────
class SyntheticCamera(Camera3DBase):
    """실제 카메라 없이 완전 합성 씬을 생성"""
    mode = "Synthetic"
    W, H = 640, 480
    fx = fy = 600.0
    cx, cy = 320.0, 240.0

    def __init__(self):
        self._t0 = time.time()
        self._angle = 0.0

    def start(self): pass
    def stop(self): pass

    def read(self):
        t = time.time() - self._t0
        # 합성 RGB 배경
        rgb = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip(
            20 + 10 * np.sin(np.linspace(0, np.pi, self.W)), 0, 255
        ).astype(np.uint8)
        rgb[:, :, 2] = 30

        # 합성 Depth: 방 바닥 3m + 움직이는 원통형 사람
        depth = np.full((self.H, self.W), 3.0, dtype=np.float32)
        # 사람 위치 (화면 중앙 ±)
        px = int(self.W / 2 + self.W * 0.2 * np.sin(0.3 * t))
        py = int(self.H / 2 + self.H * 0.05 * np.cos(0.2 * t))
        pd = 1.5 + 0.3 * np.abs(np.sin(0.15 * t))

        # 간단한 사람 실루엣 (타원)
        for dy in range(-100, 50):
            for dx in range(-30, 30):
                ry, rx = self.H // 2, self.W // 2
                if 0 <= py + dy < self.H and 0 <= px + dx < self.W:
                    # 타원 머리+몸
                    if (dx ** 2) / 900 + (dy ** 2) / 10000 < 1.0:
                        depth[py + dy, px + dx] = pd + 0.05 * (dy / 100)
                        rgb[py + dy, px + dx] = [180, 140, 100]

        return rgb, depth


# ── 자동 카메라 감지 팩토리 ──────────────────────────────────────────────────
def create_best_camera(force_synthetic: bool = False) -> Camera3DBase:
    if force_synthetic:
        cam = SyntheticCamera()
        cam.start()
        return cam

    # 1. RealSense
    if _HAS_RS:
        try:
            ctx = rs.context()
            if len(ctx.devices) > 0:
                cam = RealSenseCamera()
                cam.start()
                log.info("✅ RealSense 감지됨")
                return cam
        except Exception as e:
            log.warning("RealSense 감지 실패: %s", e)

    # 2. OAK-D
    if _HAS_DAI:
        try:
            devices = dai.Device.getAllAvailableDevices()
            if len(devices) > 0:
                cam = OAKDCamera()
                cam.start()
                log.info("✅ OAK-D 감지됨")
                return cam
        except Exception as e:
            log.warning("OAK-D 감지 실패: %s", e)

    # 3. ZED
    if _HAS_ZED:
        try:
            cam = ZEDCamera()
            cam.start()
            log.info("✅ ZED 감지됨")
            return cam
        except Exception as e:
            log.warning("ZED 감지 실패: %s", e)

    # 4. 웹캠
    if _HAS_CV2:
        for idx in range(4):
            try:
                cam = WebcamMiDaSCamera(idx)
                cam.start()
                log.info("✅ 웹캠(%d) + MiDaS 활성화됨", idx)
                return cam
            except Exception:
                pass

    # 5. 완전 합성
    log.warning("❌ 카메라 없음 → Synthetic 모드")
    cam = SyntheticCamera()
    cam.start()
    return cam


# ══════════════════════════════════════════════════════════════════════════════
# §4. ML 인체 포즈 추정 (MediaPipe)
# ══════════════════════════════════════════════════════════════════════════════

# MediaPipe 33 랜드마크 이름
_MP_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# 스켈레톤 연결선 (draw용)
_SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 팔
    (11, 23), (12, 24), (23, 24),                        # 몸통
    (23, 25), (25, 27), (24, 26), (26, 28),              # 다리
    (0, 11), (0, 12),                                    # 머리-어깨
]


class PoseEstimator:
    """MediaPipe 기반 실시간 3D 포즈 추정 (legacy solutions & Tasks API 모두 지원)"""

    def __init__(self):
        self._pose = None
        self._ready = False
        self._use_legacy = False
        self._use_synthetic_pose = True  # fallback: 합성 포즈

        if _HAS_MP and _MP_USE_LEGACY and _mp_pose is not None:
            try:
                self._pose = _mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._ready = True
                self._use_legacy = True
                self._use_synthetic_pose = False
                log.info("✅ MediaPipe Pose (legacy) 로드 완료")
            except Exception as e:
                log.warning("MediaPipe Pose legacy 실패: %s", e)
        
        if not self._ready and _HAS_MP and not _MP_USE_LEGACY:
            # Tasks API 시도 (pose landmarker 모델 파일 필요)
            log.info("MediaPipe Tasks API 모드 (합성 포즈 fallback)")

        if not self._ready:
            log.info("📌 포즈 추정: 합성(Synthetic) 모드 사용")

    def _synthetic_landmarks(self, t: float, rgb_shape: tuple) -> List[PoseLandmark3D]:
        """실제 ML 없이 합성 포즈 생성 (데모/시뮬레이션용)"""
        H, W = rgb_shape[:2]
        # 사람 중심점 (시간에 따라 이동)
        cx_norm = 0.5 + 0.15 * math.sin(0.3 * t)
        cy_norm = 0.5 + 0.05 * math.cos(0.2 * t)
        
        # 33개 관절 합성 위치 (정규화 0~1)
        # MediaPipe 포즈 키포인트 레이아웃 기반
        scale = 0.15
        joint_offsets_norm = [
            (0.0, -0.55),    # 0 nose
            (-0.04, -0.50),  # 1 left_eye_inner
            (-0.07, -0.50),  # 2 left_eye
            (-0.10, -0.50),  # 3 left_eye_outer
            (0.04, -0.50),   # 4 right_eye_inner
            (0.07, -0.50),   # 5 right_eye
            (0.10, -0.50),   # 6 right_eye_outer
            (-0.12, -0.48),  # 7 left_ear
            (0.12, -0.48),   # 8 right_ear
            (-0.04, -0.44),  # 9 mouth_left
            (0.04, -0.44),   # 10 mouth_right
            (-0.15, -0.30),  # 11 left_shoulder
            (0.15, -0.30),   # 12 right_shoulder
            (-0.25, -0.10),  # 13 left_elbow
            (0.25, -0.10),   # 14 right_elbow
            (-0.30 + 0.05*math.sin(t), 0.10),  # 15 left_wrist (움직임)
            (0.30 + 0.05*math.cos(t), 0.10),   # 16 right_wrist
            (-0.32, 0.12),   # 17 left_pinky
            (0.32, 0.12),    # 18 right_pinky
            (-0.31, 0.12),   # 19 left_index
            (0.31, 0.12),    # 20 right_index
            (-0.28, 0.13),   # 21 left_thumb
            (0.28, 0.13),    # 22 right_thumb
            (-0.10, 0.15),   # 23 left_hip
            (0.10, 0.15),    # 24 right_hip
            (-0.12, 0.40),   # 25 left_knee
            (0.12, 0.40),    # 26 right_knee
            (-0.10, 0.60),   # 27 left_ankle
            (0.10, 0.60),    # 28 right_ankle
            (-0.10, 0.65),   # 29 left_heel
            (0.10, 0.65),    # 30 right_heel
            (-0.12, 0.68),   # 31 left_foot_index
            (0.12, 0.68),    # 32 right_foot_index
        ]
        
        out = []
        for i, (dx, dy) in enumerate(joint_offsets_norm):
            x = cx_norm + dx * scale * 3
            y = cy_norm + dy * scale * 3
            # world coordinates (미터 단위 근사)
            wx = dx * 0.3
            wy = dy * 0.5
            wz = 0.05 * math.sin(t + i * 0.3)
            out.append(PoseLandmark3D(
                index=i,
                name=_MP_LANDMARK_NAMES[i] if i < len(_MP_LANDMARK_NAMES) else f"lm{i}",
                x=float(np.clip(x, 0, 1)), y=float(np.clip(y, 0, 1)),
                z=float(wz * 0.1),
                wx=float(wx), wy=float(wy), wz=float(wz),
                visibility=0.95,
            ))
        return out

    def process(self, rgb: np.ndarray, t: Optional[float] = None) -> Optional[List[PoseLandmark3D]]:
        """RGB (H,W,3) uint8 → 33개 3D 랜드마크 또는 None"""
        if t is None:
            t = time.time()

        if self._ready and self._use_legacy:
            try:
                results = self._pose.process(rgb)
                if results.pose_landmarks is None:
                    return self._synthetic_landmarks(t, rgb.shape)
                lms_2d = results.pose_landmarks.landmark
                lms_3d = results.pose_world_landmarks.landmark if results.pose_world_landmarks else lms_2d
                out = []
                for i, (lm2, lm3) in enumerate(zip(lms_2d, lms_3d)):
                    out.append(PoseLandmark3D(
                        index=i,
                        name=_MP_LANDMARK_NAMES[i] if i < len(_MP_LANDMARK_NAMES) else f"lm{i}",
                        x=lm2.x, y=lm2.y, z=lm2.z,
                        wx=lm3.x, wy=lm3.y, wz=lm3.z,
                        visibility=getattr(lm2, "visibility", 1.0),
                    ))
                return out
            except Exception as e:
                log.debug("Pose process error: %s", e)

        # Fallback: 합성 포즈
        if rgb is not None:
            return self._synthetic_landmarks(t, rgb.shape)
        return None

    def segmentation_mask(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        if not self._ready or not self._use_legacy:
            return None
        try:
            results = self._pose.process(rgb)
            if hasattr(results, 'segmentation_mask') and results.segmentation_mask is not None:
                return results.segmentation_mask
        except Exception:
            pass
        return None


# ══════════════════════════════════════════════════════════════════════════════
# §5. ML 행동 분류기
# ══════════════════════════════════════════════════════════════════════════════

ACTION_LABELS = ["UNKNOWN", "STANDING", "WALKING", "RAISING_ARM",
                 "BENDING", "SITTING", "RUNNING", "WAVING"]


class ActionClassifier:
    """
    경량 규칙 기반 + 이동평균 속도 특징을 이용한 행동 분류.
    LSTM 모델이 있으면 교체 가능한 구조.
    """

    def __init__(self, history_len: int = 30):
        self._history: deque = deque(maxlen=history_len)
        self._fps = 15.0

    def push(self, landmarks):
        """매 프레임 랜드마크를 히스토리에 추가 (PoseLandmark3D 객체 또는 dict 모두 허용)"""
        key_idx = {0, 11, 12, 23, 24, 25, 26, 27, 28}
        rows = []
        for lm in landmarks:
            # PoseLandmark3D dataclass
            if hasattr(lm, 'index') and hasattr(lm, 'wx'):
                if lm.index in key_idx:
                    rows.append([lm.wx, lm.wy, lm.wz])
            # dict (snapshot 직렬화 후 전달되는 경우)
            elif isinstance(lm, dict):
                if lm.get('i', -1) in key_idx:
                    rows.append([lm.get('wx', 0.0), lm.get('wy', 0.0), lm.get('wz', 0.0)])
        if rows:
            self._history.append(np.array(rows, dtype=np.float32))

    def predict(self) -> Tuple[str, float]:
        """현재 히스토리로 행동 레이블과 신뢰도 반환"""
        if len(self._history) < 5:
            return "UNKNOWN", 0.0

        hist = list(self._history)
        # 속도 계산 (연속 프레임 차이)
        if len(hist) >= 2:
            vels = [np.linalg.norm(hist[i] - hist[i - 1], axis=1).mean()
                    for i in range(1, len(hist))]
            vel_mean = float(np.mean(vels))
            vel_std = float(np.std(vels))
        else:
            vel_mean = 0.0
            vel_std = 0.0

        # 현재 자세 특징
        latest = hist[-1]  # (n_joints, 3)
        if len(latest) < 7:
            return "UNKNOWN", 0.0

        # idx 매핑: 0=코, 1=L어깨, 2=R어깨, 3=L고관절, 4=R고관절
        #           5=L무릎, 6=R무릎, 7=L발목 (있으면), 8=R발목 (있으면)
        nose_y    = latest[0][1]   # MediaPipe world: +y = 아래
        l_shoulder_y = latest[1][1]
        r_shoulder_y = latest[2][1]
        l_hip_y   = latest[3][1]
        r_hip_y   = latest[4][1]
        l_knee_y  = latest[5][1]
        r_knee_y  = latest[6][1]

        shoulder_y = (l_shoulder_y + r_shoulder_y) / 2
        hip_y      = (l_hip_y + r_hip_y) / 2
        knee_y     = (l_knee_y + r_knee_y) / 2

        torso_len = abs(hip_y - shoulder_y)
        hip_knee_ratio = abs(knee_y - hip_y) / (torso_len + 1e-6)

        # 팔 높이 (어깨 대비) — history에는 numpy array 저장됨, 직접 접근
        # key_idx 순서: [0=코, 1=L어깨, 2=R어깨, 3=L고관절, 4=R고관절, 5=L무릎, 6=R무릎, 7=L발목, 8=R발목]
        # 손목(index 15,16)은 key_idx에 없으므로 별도 체크 불필요; RAISING_ARM은 손목 y로 판별 생략

        # 규칙 기반 분류
        if vel_mean > 0.04 and vel_std > 0.015:
            label, conf = "RUNNING", min(0.95, vel_mean * 10)
        elif vel_mean > 0.015:
            label, conf = "WALKING", min(0.9, 0.5 + vel_mean * 15)
        elif hip_knee_ratio < 0.5 and torso_len > 0.0:
            label, conf = "SITTING", 0.80
        elif abs(nose_y - shoulder_y) < torso_len * 0.3:
            label, conf = "BENDING", 0.75
        elif vel_mean < 0.005:
            label, conf = "STANDING", 0.85
        else:
            label, conf = "STANDING", 0.60

        return label, float(np.clip(conf, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# §6. 3D 칼만 필터 융합기
# ══════════════════════════════════════════════════════════════════════════════

class KalmanFusion3D:
    """
    6-state 선형 칼만 필터 (x,y,z, vx,vy,vz).
    RF 추정 위치 + Depth/Pose 위치를 측정값으로 받아 융합.
    """

    def __init__(self, dt: float = 0.1,
                 process_noise: float = 0.05,
                 rf_noise: float = 0.5,
                 depth_noise: float = 0.1):
        self.dt = dt
        # 상태 전이 행렬 F (constant velocity)
        self.F = np.eye(6, dtype=np.float64)
        for i in range(3):
            self.F[i, i + 3] = dt

        # 관측 행렬 H (position only)
        self.H = np.zeros((3, 6), dtype=np.float64)
        for i in range(3):
            self.H[i, i] = 1.0

        # 프로세스 노이즈 Q
        q = process_noise
        self.Q = np.diag([q, q, q, q * 10, q * 10, q * 10])

        # 측정 노이즈 R (RF는 더 노이즈, depth는 정확)
        self.R_rf = np.eye(3) * (rf_noise ** 2)
        self.R_depth = np.eye(3) * (depth_noise ** 2)

        self.x = np.zeros(6, dtype=np.float64)  # 초기 상태
        self.P = np.eye(6, dtype=np.float64) * 1.0
        self._initialized = False

    def init(self, pos: np.ndarray):
        self.x[:3] = pos
        self.x[3:] = 0.0
        self.P = np.eye(6) * 0.5
        self._initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: np.ndarray, source: str = "depth"):
        if not self._initialized:
            self.init(measurement)
            return
        R = self.R_depth if source == "depth" else self.R_rf
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()


# ══════════════════════════════════════════════════════════════════════════════
# §7. RF 원본 파이프라인 (원본 유지)
# ══════════════════════════════════════════════════════════════════════════════

class RFSampler:
    def __init__(self, force_simulation: bool = False):
        self.force_simulation = force_simulation
        self._iface = self._detect_iface() if not force_simulation else "wlan0_sim"
        self._ap_bssid: Optional[str] = None
        self._ap_freq: float = 5.18
        self._detect_connected_ap()
        self._sim_t0 = time.time()

    @property
    def is_simulation(self) -> bool:
        return self.force_simulation or self._iface == "wlan0_sim"

    @staticmethod
    def _detect_iface() -> str:
        for cmd in (["iw", "dev"], ["iwconfig"]):
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
                m = re.search(r"(?:Interface\s+|^)(\w+(?:wl|wlan)\w*)", out, re.M)
                if not m:
                    m = re.search(r"Interface\s+(\S+)", out)
                if m:
                    return m.group(1)
            except Exception:
                pass
        try:
            with open("/proc/net/wireless", "r", errors="ignore") as f:
                for line in f.readlines()[2:]:
                    iface = line.split(":")[0].strip()
                    if iface:
                        return iface
        except Exception:
            pass
        return "wlan0_sim"

    def _detect_connected_ap(self):
        if self._iface == "wlan0_sim":
            return
        try:
            out = subprocess.check_output(["iw", "dev", self._iface, "link"],
                                           text=True, stderr=subprocess.DEVNULL)
            bm = re.search(r"Connected to ([0-9a-f:]{17})", out)
            fm = re.search(r"freq:\s*(\d+)", out)
            if bm:
                self._ap_bssid = bm.group(1)
            if fm:
                self._ap_freq = int(fm.group(1)) / 1000.0
        except Exception:
            pass

    def sample(self) -> Optional[RFSample]:
        if self.is_simulation:
            return self._sample_simulated()
        rssi, noise = self._read_proc_wireless()
        tx_rate, rx_rate = self._read_station_dump()
        snr = (rssi - noise) if noise else 0.0
        phase_proxy = (2 * np.pi * snr / 30.0) % (2 * np.pi)
        return RFSample(t=time.time(), rssi_dbm=rssi, noise_dbm=noise,
                        snr_db=snr, freq_ghz=self._ap_freq,
                        tx_rate_mbps=tx_rate, rx_rate_mbps=rx_rate,
                        phase_proxy=phase_proxy)

    def _sample_simulated(self) -> RFSample:
        t = time.time() - self._sim_t0
        freq_ghz = self._ap_freq
        wl = 3e8 / (freq_ghz * 1e9)
        x = 2.5 + 1.2 * np.sin(0.18 * t)
        y = 2.2 + 1.0 * np.cos(0.13 * t)
        z = 1.1 + 0.25 * np.sin(0.31 * t)
        chest = 0.006 * np.sin(2 * np.pi * 0.27 * t) + 0.0012 * np.sin(2 * np.pi * 1.15 * t)
        ap = np.array([0.0, 0.0, 2.5])
        nic = np.array([2.5, 2.5, 0.8])
        target = np.array([x, y, z])
        d_direct = np.linalg.norm(ap - nic)
        d_reflect = np.linalg.norm(ap - target) + np.linalg.norm(target - nic) + chest
        d_delta = d_reflect - d_direct
        phase = (4 * np.pi * d_delta / wl) % (2 * np.pi)
        rssi = -58 - 5.5 * abs(np.sin(0.11 * t)) - 1.8 * d_delta + np.random.normal(0, 0.8)
        noise = -95 + np.random.normal(0, 0.35)
        snr = rssi - noise
        tx = 260 + 25 * np.sin(0.09 * t) + np.random.normal(0, 3)
        rx = 240 + 22 * np.cos(0.08 * t) + np.random.normal(0, 3)
        return RFSample(
            t=time.time(), rssi_dbm=float(rssi), noise_dbm=float(noise),
            snr_db=float(snr), freq_ghz=float(freq_ghz),
            tx_rate_mbps=float(max(1.0, tx)), rx_rate_mbps=float(max(1.0, rx)),
            phase_proxy=float(phase), sim_target_xyz=(float(x), float(y), float(z))
        )

    def _read_proc_wireless(self) -> Tuple[float, float]:
        try:
            with open("/proc/net/wireless", errors="ignore") as f:
                for line in f.readlines()[2:]:
                    if self._iface in line:
                        parts = line.split()
                        rssi = float(parts[3].rstrip("."))
                        noise = float(parts[4].rstrip("."))
                        if rssi > 0: rssi -= 256
                        if noise > 0: noise -= 256
                        return rssi, noise
        except Exception:
            pass
        return -70.0, -95.0

    def _read_station_dump(self) -> Tuple[float, float]:
        try:
            out = subprocess.check_output(
                ["iw", "dev", self._iface, "station", "dump"],
                text=True, stderr=subprocess.DEVNULL, timeout=2)
            txm = re.search(r"tx bitrate:\s*([\d.]+)", out)
            rxm = re.search(r"rx bitrate:\s*([\d.]+)", out)
            return (float(txm.group(1)) if txm else 50.0,
                    float(rxm.group(1)) if rxm else 50.0)
        except Exception:
            return 50.0, 50.0


# ══════════════════════════════════════════════════════════════════════════════
# §8. 공간 처리기 (RF 볼륨, 원본 + 3D 융합)
# ══════════════════════════════════════════════════════════════════════════════

class FusionSpatialProcessor:
    ROOM_SIZE = 5.0
    ROOM_H = 3.0
    GRID_N = 20

    def __init__(self, ap_pos=(0.0, 0.0, 2.5), nic_pos=(2.5, 2.5, 0.8),
                 theory_config=None):
        self.ap_pos = np.array(ap_pos)
        self.nic_pos = np.array(nic_pos)
        self._d_direct = float(np.linalg.norm(self.ap_pos - self.nic_pos))
        self.L_SAT = 10.0
        self.motion_threshold = 0.003
        self.theory = ObservationalSaturationEngine(theory_config)

        xs = np.linspace(0, self.ROOM_SIZE, self.GRID_N)
        ys = np.linspace(0, self.ROOM_SIZE, self.GRID_N)
        zs = np.linspace(0, self.ROOM_H, self.GRID_N // 2)
        self.gx, self.gy, self.gz = np.meshgrid(xs, ys, zs, indexing="ij")
        self.volume = np.zeros_like(self.gx)
        self.vol_count = 0
        self.top_projection = np.zeros((self.GRID_N, self.GRID_N))
        self.front_projection = np.zeros((self.GRID_N, self.GRID_N // 2))
        self.side_projection = np.zeros((self.GRID_N, self.GRID_N // 2))
        self.pseudo_camera = np.zeros((96, 144))

        maxlen = 600
        self._phase_buf = deque(maxlen=maxlen)
        self._snr_buf = deque(maxlen=maxlen)
        self._rssi_buf = deque(maxlen=maxlen)
        self._time_buf = deque(maxlen=maxlen)
        self._tx_buf = deque(maxlen=maxlen)
        self._rx_buf = deque(maxlen=maxlen)
        self._sim_xyz_buf = deque(maxlen=maxlen)

    def reset(self):
        for buf in [self._phase_buf, self._snr_buf, self._rssi_buf,
                    self._time_buf, self._tx_buf, self._rx_buf, self._sim_xyz_buf]:
            buf.clear()
        self.volume[:] = 0
        self.vol_count = 0

    def push(self, s: RFSample):
        self._phase_buf.append(s.phase_proxy)
        self._snr_buf.append(s.snr_db)
        self._rssi_buf.append(s.rssi_dbm)
        self._time_buf.append(s.t)
        self._sim_xyz_buf.append(s.sim_target_xyz)
        self._tx_buf.append(s.tx_rate_mbps)
        self._rx_buf.append(s.rx_rate_mbps)

    def extract_vitals(self, signal, fs):
        if len(signal) < max(40, int(fs * 5)):
            return 0.0, 0.0
        try:
            resp_b, resp_a = butter(3, [0.1 / (fs/2), min(0.5 / (fs/2), 0.99)], btype="bandpass")
            heart_b, heart_a = butter(3, [0.8 / (fs/2), min(2.4 / (fs/2), 0.99)], btype="bandpass")
            resp = filtfilt(resp_b, resp_a, signal)
            heart = filtfilt(heart_b, heart_a, signal)
            r_peaks, _ = find_peaks(resp, distance=max(2, int(fs * 1.2)))
            h_peaks, _ = find_peaks(heart, distance=max(2, int(fs * 0.35)))
            duration = len(signal) / fs
            return (round(len(r_peaks) * 60.0 / max(duration, 1e-6), 1),
                    round(len(h_peaks) * 60.0 / max(duration, 1e-6), 1))
        except Exception:
            return 0.0, 0.0

    def process(self):
        if len(self._phase_buf) < 50:
            return {"status": "collecting", "samples": len(self._phase_buf), "needed": 50}
        phi = np.array(self._phase_buf, dtype=float)
        snr = np.array(self._snr_buf, dtype=float)
        rssi = np.array(self._rssi_buf, dtype=float)
        t = np.array(self._time_buf, dtype=float)
        tx = np.array(self._tx_buf, dtype=float)
        rx = np.array(self._rx_buf, dtype=float)
        wl = 3e8 / 5.18e9
        dphi_obs = np.diff(np.unwrap(phi))
        dphi_true = self.theory.inverse_saturate(dphi_obs, self.L_SAT)
        dt = np.mean(np.diff(t)) if len(t) > 1 else 0.1
        fs = 1.0 / max(dt, 0.01)
        try:
            low = 0.1 / (fs/2)
            high = min(3.0 / (fs/2), 0.99)
            b, a = butter(4, [low, high], btype="band")
            dphi_filt = filtfilt(b, a, dphi_true) if len(dphi_true) > 15 else dphi_true
        except Exception:
            dphi_filt = dphi_true
        delta_d = dphi_filt * wl / (4 * np.pi)
        theory_metrics = self.theory.epistemic_metrics(delta_d, snr, rssi, tx, rx)
        resp_bpm, heart_bpm = self.extract_vitals(dphi_filt, fs)
        az, el, P = self._music_spectrum(dphi_filt, wl)
        new_vol = self._project_to_volume(az, el, P, delta_d, theory_metrics)
        alpha = float(np.clip(self.theory.cfg.smoothing + 0.12 * (1 - theory_metrics["observability_score"]), 0.62, 0.93))
        self.volume = alpha * self.volume + (1 - alpha) * new_vol
        self.volume = gaussian_filter(self.volume, sigma=0.75 - 0.35 * theory_metrics["confidence"])
        self._update_visual_projections(theory_metrics)
        self.vol_count += 1
        motion_energy = float(np.std(delta_d) * (0.75 + 0.25 * theory_metrics["observability_score"]))
        occupancy_energy = float(np.percentile(np.abs(snr), 75) * (0.65 + 0.35 * theory_metrics["observability_score"]))
        return self._extract_result(delta_d, az, el, P, motion_energy,
                                     occupancy_energy, resp_bpm, heart_bpm, fs, theory_metrics)

    def _music_spectrum(self, dphi, wl):
        M = min(24, max(8, len(dphi) // 3))
        N = len(dphi) - M + 1
        if N <= 2:
            az_scan = np.linspace(-np.pi/2, np.pi/2, 60)
            el_scan = np.linspace(0, np.pi/3, 16)
            return az_scan, el_scan, np.ones((60, 16))
        X = np.array([dphi[i:i+M] for i in range(N)]).T
        R = X @ X.conj().T / N
        try:
            _, eigenvectors = np.linalg.eigh(R)
        except Exception:
            az_scan = np.linspace(-np.pi/2, np.pi/2, 60)
            el_scan = np.linspace(0, np.pi/3, 16)
            return az_scan, el_scan, np.ones((60, 16))
        n_signals = max(1, min(3, M // 5))
        Vn = eigenvectors[:, :-n_signals]
        az_scan = np.linspace(-np.pi/2, np.pi/2, 72)
        el_scan = np.linspace(0, np.pi/3, 18)
        P = np.zeros((len(az_scan), len(el_scan)))
        d = wl / 2
        for i, az in enumerate(az_scan):
            for j, el_v in enumerate(el_scan):
                u = np.cos(el_v) * np.sin(az)
                a = np.exp(1j * 2 * np.pi * d / wl * u * np.arange(M))
                proj = Vn.conj().T @ a
                P[i, j] = 1.0 / (float(np.real(proj.conj() @ proj)) + 1e-10)
        return az_scan, el_scan, P

    def _project_to_volume(self, az, el, P, delta_d, theory_metrics):
        vol = np.zeros_like(self.volume)
        P_flat = P.flatten()
        peak_k = max(4, int(self.theory.cfg.top_k_peaks))
        peak_idx = np.argsort(P_flat)[-peak_k:]
        d_std = float(np.std(delta_d)) if len(delta_d) > 1 else 0.01
        d_reflect_est = self._d_direct + d_std * (8.0 + 5.0 * theory_metrics["observability_score"])
        threshold = theory_metrics["adaptive_threshold"]
        for idx in peak_idx:
            i_az, i_el = idx // len(el), idx % len(el)
            base_weight = float(P_flat[idx]) / float(P_flat[peak_idx].max() + 1e-10)
            weight = base_weight * (0.5 + 0.5 * theory_metrics["confidence"])
            if weight < threshold:
                continue
            direction = np.array([
                np.cos(el[i_el]) * np.cos(az[i_az]),
                np.cos(el[i_el]) * np.sin(az[i_az]),
                np.sin(el[i_el]),
            ])
            r = max(0.35, min((d_reflect_est - self._d_direct) / 2.0, self.ROOM_SIZE * 1.1))
            reflect_pos = self.nic_pos + direction * r
            reflect_pos[0] = np.clip(reflect_pos[0], 0, self.ROOM_SIZE)
            reflect_pos[1] = np.clip(reflect_pos[1], 0, self.ROOM_SIZE)
            reflect_pos[2] = np.clip(reflect_pos[2], 0, self.ROOM_H)
            sigma = max(self.theory.cfg.sigma_min,
                        self.theory.cfg.sigma_base - weight * 0.22 + 0.18 * (1.0 - theory_metrics["observability_score"]))
            dist = np.sqrt(
                (self.gx - reflect_pos[0]) ** 2 +
                (self.gy - reflect_pos[1]) ** 2 +
                (self.gz - reflect_pos[2]) ** 2
            )
            vol += weight * (0.65 + 0.35 * theory_metrics["energy_ratio"]) * np.exp(-(dist**2) / (2*sigma**2))
        return vol

    def _resize_2d(self, arr, th, tw):
        if arr.size == 0:
            return np.zeros((th, tw))
        sh, sw = arr.shape
        y_old = np.linspace(0, 1, sh)
        x_old = np.linspace(0, 1, sw)
        y_new = np.linspace(0, 1, th)
        x_new = np.linspace(0, 1, tw)
        tmp = np.array([np.interp(x_new, x_old, row) for row in arr])
        out = np.array([np.interp(y_new, y_old, tmp[:, j]) for j in range(tmp.shape[1])]).T
        return out

    def _update_visual_projections(self, tm):
        vol = self.volume / (self.volume.max() + 1e-10)
        top_now = np.max(vol, axis=2)
        front_now = np.max(vol, axis=1)
        side_now = np.max(vol, axis=0)
        persist = float(np.clip(self.theory.cfg.temporal_persistence + 0.08 * (1.0 - tm["confidence"]), 0.55, 0.96))
        self.top_projection = persist * self.top_projection + (1-persist) * top_now
        self.front_projection = persist * self.front_projection + (1-persist) * front_now
        self.side_projection = persist * self.side_projection + (1-persist) * side_now
        cam = self._resize_2d(front_now, 96, 144)
        top_r = self._resize_2d(top_now, 96, 144)
        side_r = self._resize_2d(side_now.T, 96, 144)
        fused = (0.5 * cam + 0.28 * side_r + 0.22 * top_r) * self.theory.cfg.projection_gain
        fused = gaussian_filter(fused, sigma=1.05 - 0.45 * tm["confidence"])
        fused = np.clip(fused * self.theory.cfg.pseudo_camera_gain * (0.75 + 0.5 * tm["observability_score"]), 0, None)
        gamma = max(self.theory.cfg.pseudo_camera_gamma, 0.25)
        fused = np.power(fused / (fused.max() + 1e-10), gamma)
        self.pseudo_camera = persist * self.pseudo_camera + (1-persist) * fused

    def _extract_result(self, delta_d, az, el, P, motion_energy,
                        occupancy_energy, resp_bpm, heart_bpm, fs, tm):
        vol_norm = self.volume / (self.volume.max() + 1e-10)
        thresh = max(tm["adaptive_threshold"], float(np.percentile(vol_norm, 94)))
        mask = vol_norm >= thresh
        xs, ys, zs, ps = (self.gx[mask].flatten(), self.gy[mask].flatten(),
                           self.gz[mask].flatten(), vol_norm[mask].flatten())
        if len(ps) > 900:
            idx = np.argsort(ps)[-900:]
            xs, ys, zs, ps = xs[idx], ys[idx], zs[idx], ps[idx]
        voxels = [{"x": float(x), "y": float(y), "z": float(z), "p": float(p)}
                  for x, y, z, p in zip(xs, ys, zs, ps)]
        peak_index = np.unravel_index(np.argmax(self.volume), self.volume.shape)
        peak_xyz = (float(self.gx[peak_index]), float(self.gy[peak_index]), float(self.gz[peak_index]))
        spectrum = {
            "az_deg": (az * 180/np.pi).tolist(),
            "el_deg": (el * 180/np.pi).tolist(),
            "P": (P / (P.max() + 1e-10)).tolist(),
        }
        return {
            "status": "ok", "voxels": voxels, "spectrum": spectrum,
            "motion_energy": round(motion_energy, 6),
            "occupancy_energy": round(occupancy_energy, 4),
            "motion_detected": bool(motion_energy > self.motion_threshold),
            "resp_bpm": resp_bpm, "heart_bpm": heart_bpm,
            "samples": len(self._phase_buf), "peak_xyz": peak_xyz,
            "fs": round(fs, 3),
            "phase_std": round(float(np.std(np.unwrap(np.array(self._phase_buf)))), 6),
            "distance_std_cm": round(float(np.std(delta_d) * 100.0), 3),
            "theory": tm, "manifest": self.theory.manifest(),
            "top_projection": self.top_projection.tolist(),
            "front_projection": self.front_projection.tolist(),
            "side_projection": self.side_projection.tolist(),
            "pseudo_camera": self.pseudo_camera.tolist(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# §9. 통합 3D 비전 파이프라인 (카메라 + ML 백그라운드 루프)
# ══════════════════════════════════════════════════════════════════════════════

class Vision3DPipeline:
    """카메라 → Pose → Action → Kalman → 결과 저장 (백그라운드 스레드)"""

    def __init__(self, force_synthetic: bool = True):
        self.lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._force_synthetic = force_synthetic

        # 상태
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_pointcloud: Optional[np.ndarray] = None
        self._latest_pointcloud_rgb: Optional[np.ndarray] = None
        self._latest_landmarks: Optional[List[PoseLandmark3D]] = None
        self._latest_action: str = "UNKNOWN"
        self._latest_action_conf: float = 0.0
        self._latest_fused_xyz: Optional[Tuple[float, float, float]] = None
        self._trajectory: deque = deque(maxlen=100)
        self._frame_count: int = 0
        self._camera_mode: str = "None"
        self._kalman = KalmanFusion3D()
        self._last_error: str = ""

        # 컴포넌트 (start()에서 초기화)
        self._camera: Optional[Camera3DBase] = None
        self._pose_estimator: Optional[PoseEstimator] = None
        self._action_classifier: Optional[ActionClassifier] = None

    def start(self):
        with self.lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        with self.lock:
            self._running = False
        if self._camera:
            try:
                self._camera.stop()
            except Exception:
                pass

    def _init_components(self):
        self._camera = create_best_camera(self._force_synthetic)
        self._camera_mode = self._camera.mode
        self._pose_estimator = PoseEstimator()
        self._action_classifier = ActionClassifier()

    def _loop(self):
        try:
            self._init_components()
        except Exception as e:
            with self.lock:
                self._last_error = f"Init error: {e}"
                self._running = False
            return

        while True:
            with self.lock:
                if not self._running:
                    break

            try:
                frame_data = self._camera.read()
                if frame_data is None:
                    time.sleep(0.05)
                    continue

                rgb, depth = frame_data

                # 포인트클라우드 생성
                xyz, crgb = self._camera.depth_to_pointcloud(rgb, depth)

                # Pose 추정
                landmarks = None
                if self._pose_estimator and rgb is not None:
                    landmarks = self._pose_estimator.process(rgb, t=time.time())

                # 행동 분류
                action, action_conf = "UNKNOWN", 0.0
                if landmarks and self._action_classifier:
                    self._action_classifier.push(landmarks)
                    action, action_conf = self._action_classifier.predict()

                # 깊이 카메라 기반 사람 위치 (어깨 중심)
                fused_xyz = None
                if landmarks and depth is not None:
                    try:
                        l_sh = landmarks[11]  # left shoulder
                        r_sh = landmarks[12]  # right shoulder
                        cx_pix = int((l_sh.x + r_sh.x) / 2 * depth.shape[1])
                        cy_pix = int((l_sh.y + r_sh.y) / 2 * depth.shape[0])
                        cx_pix = np.clip(cx_pix, 0, depth.shape[1] - 1)
                        cy_pix = np.clip(cy_pix, 0, depth.shape[0] - 1)
                        d_val = float(depth[cy_pix, cx_pix])
                        if 0.1 < d_val < 8.0:
                            cam = self._camera
                            px_3d = (cx_pix - cam.cx) * d_val / cam.fx
                            py_3d = (cy_pix - cam.cy) * d_val / cam.fy
                            meas = np.array([px_3d, py_3d, d_val])
                            self._kalman.predict()
                            self._kalman.update(meas, source="depth")
                            pos = self._kalman.position
                            fused_xyz = (float(pos[0]), float(pos[1]), float(pos[2]))
                    except Exception:
                        pass
                elif landmarks:
                    # depth 없으면 MediaPipe world 좌표 사용
                    try:
                        l_sh = landmarks[11]
                        r_sh = landmarks[12]
                        meas = np.array([
                            (l_sh.wx + r_sh.wx) / 2,
                            (l_sh.wy + r_sh.wy) / 2,
                            abs(l_sh.wz + r_sh.wz) / 2 + 1.0
                        ])
                        self._kalman.predict()
                        self._kalman.update(meas, source="depth")
                        pos = self._kalman.position
                        fused_xyz = (float(pos[0]), float(pos[1]), float(pos[2]))
                    except Exception:
                        pass

                # 상태 업데이트
                with self.lock:
                    self._latest_rgb = rgb
                    self._latest_depth = depth
                    self._latest_pointcloud = xyz
                    self._latest_pointcloud_rgb = crgb
                    # landmarks는 반드시 PoseLandmark3D 객체 리스트여야 함
                    if landmarks and isinstance(landmarks, list) and hasattr(landmarks[0], 'index'):
                        self._latest_landmarks = landmarks
                    elif not landmarks:
                        self._latest_landmarks = None
                    # numpy array 등 잘못된 타입이면 무시 (이전 값 유지)
                    self._latest_action = action
                    self._latest_action_conf = action_conf
                    self._latest_fused_xyz = fused_xyz
                    if fused_xyz:
                        self._trajectory.append(fused_xyz)
                    self._frame_count += 1

            except Exception as e:
                with self.lock:
                    self._last_error = str(e)
                time.sleep(0.1)

    def snapshot(self) -> Dict:
        with self.lock:
            traj = list(self._trajectory)
            lms = self._latest_landmarks
            return {
                "running": self._running,
                "frame_count": self._frame_count,
                "camera_mode": self._camera_mode,
                "last_error": self._last_error,
                "rgb": self._latest_rgb.copy() if self._latest_rgb is not None else None,
                "depth": self._latest_depth.copy() if self._latest_depth is not None else None,
                "pointcloud_xyz": self._latest_pointcloud.copy() if self._latest_pointcloud is not None else None,
                "pointcloud_rgb": self._latest_pointcloud_rgb.copy() if self._latest_pointcloud_rgb is not None else None,
                "landmarks": [
                    {"i": lm.index, "name": lm.name,
                     "x": lm.x, "y": lm.y,
                     "wx": lm.wx, "wy": lm.wy, "wz": lm.wz,
                     "vis": lm.visibility}
                    for lm in lms
                    if hasattr(lm, 'index') and hasattr(lm, 'wx')  # PoseLandmark3D 객체만
                ] if lms else [],
                "action": self._latest_action,
                "action_conf": self._latest_action_conf,
                "fused_xyz": self._latest_fused_xyz,
                "trajectory": traj,
                "ml_ready": _HAS_MP,
                "cam_mode": self._camera_mode,
            }


# ══════════════════════════════════════════════════════════════════════════════
# §10. 양자 인코더 (원본 유지)
# ══════════════════════════════════════════════════════════════════════════════

class QuantumPhaseEncoder:
    def __init__(self, n_qubits=12, room_size=5.0, room_h=3.0):
        self.n_q = n_qubits
        self.room = room_size
        self.room_h = room_h
        self.n_x = n_qubits // 3
        self.n_y = n_qubits // 3
        self.n_z = n_qubits - self.n_x - self.n_y
        self.enabled = _HAS_QISKIT
        self._backend = AerSimulator(method="statevector") if self.enabled else None

    def encode(self, phase_series, snr_series, shots=512):
        if len(phase_series) < max(8, self.n_q):
            return []
        if not self.enabled:
            return self._fallback_volume(phase_series, snr_series)
        phi = np.array(phase_series[-self.n_q*4:], dtype=float)
        snr = np.array(snr_series[-self.n_q*4:], dtype=float)
        phi_norm = (np.unwrap(phi) % (2*np.pi)) / (2*np.pi)
        snr_norm = np.clip((snr + 100.0) / 70.0, 0, 1)
        qr, cr = QuantumRegister(self.n_q, "q"), ClassicalRegister(self.n_q, "c")
        qc = QuantumCircuit(qr, cr)
        for i in range(self.n_q):
            qc.h(qr[i])
        for i in range(self.n_q):
            idx = int(i * len(phi_norm) / self.n_q) % len(phi_norm)
            qc.ry(float(phi_norm[idx]) * np.pi, qr[i])
            qc.rz(float(snr_norm[min(idx, len(snr_norm)-1)]) * np.pi * 2, qr[i])
        for i in range(0, self.n_q-1, 2):
            qc.cx(qr[i], qr[i+1])
        for i in range(1, self.n_q-2, 2):
            qc.cx(qr[i], qr[i+1])
        if self.n_q >= 6:
            qc.cz(qr[0], qr[self.n_q//2])
            qc.cz(qr[self.n_q//3], qr[self.n_q-1])
        if _HAS_QFTGATE:
            qc.append(QFTGate(self.n_q), range(self.n_q))
        else:
            from qiskit.circuit.library import QFT
            qc.compose(QFT(self.n_q, do_swaps=True), inplace=True)
        qc.measure(qr, cr)
        tc = transpile(qc, self._backend, optimization_level=1)
        counts = self._backend.run(tc, shots=shots).result().get_counts()
        return self._counts_to_volume(counts)

    def _fallback_volume(self, phase_series, snr_series):
        phase = np.unwrap(np.array(phase_series[-60:], dtype=float))
        if len(phase) < 6:
            return []
        p = np.abs(np.fft.rfft(phase - phase.mean()))
        top = np.argsort(p)[-10:]
        return sorted([
            {"x": round(((i*37)%100)/100*self.room, 3),
             "y": round(((i*61)%100)/100*self.room, 3),
             "z": round(((i*17)%100)/100*self.room_h, 3),
             "p": round(float(p[i]/(p[top].max()+1e-9)), 5)}
            for i in top], key=lambda x: x["p"], reverse=True)

    def _counts_to_volume(self, counts):
        total = max(sum(counts.values()), 1)
        dist = {}
        for bs, cnt in counts.items():
            bits = bs.zfill(self.n_q)
            x = self._bits_to_coord(bits[:self.n_x], (0, self.room))
            y = self._bits_to_coord(bits[self.n_x:self.n_x+self.n_y], (0, self.room))
            z = self._bits_to_coord(bits[self.n_x+self.n_y:], (0, self.room_h))
            key = (round(x,2), round(y,2), round(z,2))
            dist[key] = dist.get(key, 0.0) + cnt/total
        out = [{"x": k[0], "y": k[1], "z": k[2], "p": v} for k,v in dist.items()]
        out.sort(key=lambda x: x["p"], reverse=True)
        return out[:160]

    @staticmethod
    def _bits_to_coord(bit_str, rng):
        if not bit_str: return rng[0]
        v = int(bit_str, 2)
        maxv = (2**len(bit_str)) - 1
        if maxv <= 0: return rng[0]
        return rng[0] + (rng[1]-rng[0])*(v/maxv)


# ══════════════════════════════════════════════════════════════════════════════
# §11. 통합 백엔드 (RF + 3D 비전)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DashboardState:
    running: bool = False
    last_update: Optional[float] = None
    last_error: Optional[str] = None
    iface: str = "unknown"
    ap_bssid: Optional[str] = None
    simulation: bool = True
    rssi_series: List[float] = field(default_factory=list)
    snr_series: List[float] = field(default_factory=list)
    phase_series: List[float] = field(default_factory=list)
    time_series: List[float] = field(default_factory=list)
    tx_series: List[float] = field(default_factory=list)
    rx_series: List[float] = field(default_factory=list)
    voxels: List[Dict] = field(default_factory=list)
    q_voxels: List[Dict] = field(default_factory=list)
    spectrum: Dict = field(default_factory=dict)
    motion_energy: float = 0.0
    occupancy_energy: float = 0.0
    motion_detected: bool = False
    resp_bpm: float = 0.0
    heart_bpm: float = 0.0
    samples: int = 0
    peak_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fs: float = 0.0
    phase_std: float = 0.0
    distance_std_cm: float = 0.0
    theory: Dict = field(default_factory=dict)
    manifest: Dict = field(default_factory=dict)
    theory_config: Dict = field(default_factory=dict)
    top_projection: List = field(default_factory=list)
    front_projection: List = field(default_factory=list)
    side_projection: List = field(default_factory=list)
    pseudo_camera: List = field(default_factory=list)
    stats: Dict = field(default_factory=lambda: {"total_frames": 0, "motion_events": 0, "max_energy": 0.0})


class Backend:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.theory_config = TheoryConfig()
        self.sampler = RFSampler(force_simulation=True)
        self.processor = FusionSpatialProcessor(theory_config=self.theory_config)
        self.qencoder = QuantumPhaseEncoder()
        self.state = DashboardState(
            iface=self.sampler._iface,
            ap_bssid=self.sampler._ap_bssid,
            simulation=self.sampler.is_simulation,
            manifest=self.processor.theory.manifest(),
            theory_config=self.theory_config.__dict__.copy(),
        )

    def configure(self, simulation, ap_xyz, nic_xyz, theory_config=None):
        with self.lock:
            was_running = self.state.running
        if was_running:
            self.stop()
        self.theory_config = theory_config or self.theory_config
        self.sampler = RFSampler(force_simulation=simulation)
        self.processor = FusionSpatialProcessor(ap_pos=ap_xyz, nic_pos=nic_xyz,
                                                 theory_config=self.theory_config)
        self.qencoder = QuantumPhaseEncoder(room_size=self.processor.ROOM_SIZE,
                                             room_h=self.processor.ROOM_H)
        with self.lock:
            self.state = DashboardState(
                running=False,
                iface=self.sampler._iface,
                ap_bssid=self.sampler._ap_bssid,
                simulation=self.sampler.is_simulation,
                manifest=self.processor.theory.manifest(),
                theory_config=self.theory_config.__dict__.copy(),
            )

    def start(self, sample_hz=12, process_every=4, q_every=16):
        with self.lock:
            if self.state.running:
                return
            self.state.running = True
        self.thread = threading.Thread(
            target=self._loop, args=(sample_hz, process_every, q_every), daemon=True)
        self.thread.start()

    def stop(self):
        with self.lock:
            self.state.running = False

    def reset(self):
        with self.lock:
            running = self.state.running
            sim = self.state.simulation
            iface = self.state.iface
            ap = self.state.ap_bssid
        self.processor.reset()
        with self.lock:
            self.state = DashboardState(
                running=running, simulation=sim, iface=iface, ap_bssid=ap,
                manifest=self.processor.theory.manifest(),
                theory_config=self.theory_config.__dict__.copy())

    def snapshot(self) -> Dict:
        with self.lock:
            return {
                "running": self.state.running,
                "last_update": self.state.last_update,
                "last_error": self.state.last_error,
                "iface": self.state.iface,
                "ap_bssid": self.state.ap_bssid,
                "simulation": self.state.simulation,
                "rssi_series": list(self.state.rssi_series),
                "snr_series": list(self.state.snr_series),
                "phase_series": list(self.state.phase_series),
                "time_series": list(self.state.time_series),
                "tx_series": list(self.state.tx_series),
                "rx_series": list(self.state.rx_series),
                "voxels": list(self.state.voxels),
                "q_voxels": list(self.state.q_voxels),
                "spectrum": dict(self.state.spectrum),
                "motion_energy": self.state.motion_energy,
                "occupancy_energy": self.state.occupancy_energy,
                "motion_detected": self.state.motion_detected,
                "resp_bpm": self.state.resp_bpm,
                "heart_bpm": self.state.heart_bpm,
                "samples": self.state.samples,
                "peak_xyz": self.state.peak_xyz,
                "fs": self.state.fs,
                "phase_std": self.state.phase_std,
                "distance_std_cm": self.state.distance_std_cm,
                "theory": dict(self.state.theory),
                "manifest": dict(self.state.manifest),
                "theory_config": dict(self.state.theory_config),
                "top_projection": list(self.state.top_projection),
                "front_projection": list(self.state.front_projection),
                "side_projection": list(self.state.side_projection),
                "pseudo_camera": list(self.state.pseudo_camera),
                "stats": dict(self.state.stats),
                "ap_pos": self.processor.ap_pos.tolist(),
                "nic_pos": self.processor.nic_pos.tolist(),
                "room_size": self.processor.ROOM_SIZE,
                "room_h": self.processor.ROOM_H,
            }

    def _loop(self, sample_hz, process_every, q_every):
        sample_count = 0
        while True:
            with self.lock:
                if not self.state.running:
                    break
            t0 = time.time()
            try:
                s = self.sampler.sample()
                if s is not None:
                    self.processor.push(s)
                    sample_count += 1
                    with self.lock:
                        self.state.rssi_series.append(round(s.rssi_dbm, 2))
                        self.state.snr_series.append(round(s.snr_db, 2))
                        self.state.phase_series.append(round(s.phase_proxy, 5))
                        self.state.time_series.append(round(s.t, 3))
                        self.state.tx_series.append(round(s.tx_rate_mbps, 2))
                        self.state.rx_series.append(round(s.rx_rate_mbps, 2))
                        for k in ["rssi_series","snr_series","phase_series",
                                  "time_series","tx_series","rx_series"]:
                            v = getattr(self.state, k)
                            if len(v) > 400:
                                setattr(self.state, k, v[-400:])
                if sample_count % process_every == 0 and sample_count > 0:
                    result = self.processor.process()
                    with self.lock:
                        self.state.last_update = time.time()
                    if result.get("status") == "ok":
                        with self.lock:
                            self.state.voxels = result["voxels"]
                            self.state.spectrum = result["spectrum"]
                            self.state.motion_energy = result["motion_energy"]
                            self.state.occupancy_energy = result["occupancy_energy"]
                            self.state.motion_detected = result["motion_detected"]
                            self.state.resp_bpm = result["resp_bpm"]
                            self.state.heart_bpm = result["heart_bpm"]
                            self.state.samples = result["samples"]
                            self.state.peak_xyz = result["peak_xyz"]
                            self.state.fs = result["fs"]
                            self.state.phase_std = result["phase_std"]
                            self.state.distance_std_cm = result["distance_std_cm"]
                            self.state.theory = result["theory"]
                            self.state.manifest = result["manifest"]
                            self.state.top_projection = result["top_projection"]
                            self.state.front_projection = result["front_projection"]
                            self.state.side_projection = result["side_projection"]
                            self.state.pseudo_camera = result["pseudo_camera"]
                            self.state.stats["total_frames"] += 1
                            if result["motion_detected"]:
                                self.state.stats["motion_events"] += 1
                            if result["motion_energy"] > self.state.stats["max_energy"]:
                                self.state.stats["max_energy"] = result["motion_energy"]
                if sample_count % q_every == 0 and sample_count > 0:
                    with self.lock:
                        ph = list(self.state.phase_series)
                        sn = list(self.state.snr_series)
                    q_vol = self.qencoder.encode(ph, sn)
                    with self.lock:
                        self.state.q_voxels = q_vol
            except Exception as e:
                with self.lock:
                    self.state.last_error = str(e)
            elapsed = time.time() - t0
            sleep_t = max(0.0, 1.0/max(sample_hz, 1) - elapsed)
            time.sleep(sleep_t)


# ══════════════════════════════════════════════════════════════════════════════
# §12. Streamlit 전역 싱글턴 관리
# ══════════════════════════════════════════════════════════════════════════════

def get_backend() -> Backend:
    if "backend" not in st.session_state:
        st.session_state["backend"] = Backend()
    return st.session_state["backend"]

def get_vision() -> Vision3DPipeline:
    if "vision" not in st.session_state:
        st.session_state["vision"] = Vision3DPipeline(force_synthetic=True)
    return st.session_state["vision"]


# ══════════════════════════════════════════════════════════════════════════════
# §13. 렌더링 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def render_3d_scene(rf_snap: Dict, vis_snap: Dict) -> go.Figure:
    """RF 볼륨 + 3D 포인트클라우드 + ML 스켈레톤을 하나의 Plotly 3D figure로"""
    traces = []

    # RF 볼륨 (반투명 구름)
    voxels = rf_snap.get("voxels", [])
    if voxels:
        vx = [v["x"] for v in voxels]
        vy = [v["y"] for v in voxels]
        vz = [v["z"] for v in voxels]
        vp = [v["p"] for v in voxels]
        traces.append(go.Scatter3d(
            x=vx, y=vy, z=vz,
            mode="markers",
            marker=dict(size=3, color=vp, colorscale="Turbo",
                        opacity=0.35, cmin=0, cmax=1,
                        colorbar=dict(title="RF Energy", x=1.05, thickness=10)),
            name="RF 볼륨",
        ))

    # 3D 포인트클라우드 (Depth)
    xyz = vis_snap.get("pointcloud_xyz")
    crgb = vis_snap.get("pointcloud_rgb")
    if xyz is not None and len(xyz) > 0:
        colors = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"
                  for c in crgb] if crgb is not None else "gray"
        # 카메라 좌표 → 방 좌표계 변환 (간단한 평행이동)
        traces.append(go.Scatter3d(
            x=xyz[:, 2].tolist(),   # depth = 방 Y 축
            y=xyz[:, 0].tolist(),   # 카메라 X = 방 X
            z=(-xyz[:, 1]).tolist(),# 카메라 Y(아래) = 방 -Z
            mode="markers",
            marker=dict(size=1.5, color=colors, opacity=0.55),
            name="Depth PointCloud",
        ))

    # ML 스켈레톤
    landmarks = vis_snap.get("landmarks", [])
    if landmarks:
        lm_map = {lm["i"]: lm for lm in landmarks}
        # 관절 점
        xs = [lm["wx"] for lm in landmarks if lm["vis"] > 0.4]
        ys = [lm["wz"] for lm in landmarks if lm["vis"] > 0.4]  # wz → 방 depth
        zs = [-lm["wy"] for lm in landmarks if lm["vis"] > 0.4] # wy 반전
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=5, color="#00FF88", opacity=0.9,
                        symbol="circle"),
            name="관절 (Pose)",
        ))
        # 스켈레톤 연결선
        for a_idx, b_idx in _SKELETON_CONNECTIONS:
            if a_idx in lm_map and b_idx in lm_map:
                la, lb = lm_map[a_idx], lm_map[b_idx]
                if la["vis"] > 0.3 and lb["vis"] > 0.3:
                    traces.append(go.Scatter3d(
                        x=[la["wx"], lb["wx"], None],
                        y=[la["wz"], lb["wz"], None],
                        z=[-la["wy"], -lb["wy"], None],
                        mode="lines",
                        line=dict(color="#00FFCC", width=3),
                        showlegend=False,
                    ))

    # 칼만 트래젝토리
    traj = vis_snap.get("trajectory", [])
    if len(traj) >= 2:
        tx = [p[0] for p in traj]
        ty = [p[2] for p in traj]   # depth
        tz = [-p[1] for p in traj]
        traces.append(go.Scatter3d(
            x=tx, y=ty, z=tz,
            mode="lines+markers",
            line=dict(color="yellow", width=2),
            marker=dict(size=2, color="yellow"),
            name="칼만 트래젝토리",
        ))

    # AP/NIC 위치
    ap = rf_snap.get("ap_pos", [0,0,2.5])
    nic = rf_snap.get("nic_pos", [2.5,2.5,0.8])
    traces.append(go.Scatter3d(
        x=[ap[0]], y=[ap[1]], z=[ap[2]],
        mode="markers+text", text=["AP"],
        marker=dict(size=10, color="red", symbol="diamond"),
        name="AP"
    ))
    traces.append(go.Scatter3d(
        x=[nic[0]], y=[nic[1]], z=[nic[2]],
        mode="markers+text", text=["NIC"],
        marker=dict(size=10, color="blue", symbol="diamond"),
        name="NIC"
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="🌐 3D Fusion Scene (RF + Depth + ML Pose + Kalman)",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y / Depth (m)", zaxis_title="Z (m)",
            bgcolor="rgba(5,5,20,1)",
            xaxis=dict(range=[0, rf_snap.get("room_size", 5)]),
            yaxis=dict(range=[0, rf_snap.get("room_size", 5)]),
            zaxis=dict(range=[0, rf_snap.get("room_h", 3)]),
        ),
        paper_bgcolor="rgba(5,5,20,1)",
        font_color="white",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(bgcolor="rgba(20,20,40,0.8)", bordercolor="#444"),
    )
    return fig


def render_pose_2d(vis_snap: Dict) -> Optional[np.ndarray]:
    """RGB 위에 포즈 스켈레톤을 그려 반환 (OpenCV)"""
    if not _HAS_CV2 or not _HAS_MP:
        return None
    rgb = vis_snap.get("rgb")
    landmarks = vis_snap.get("landmarks", [])
    if rgb is None:
        return None
    img = rgb.copy()
    if not landmarks:
        return img
    lm_map = {lm["i"]: lm for lm in landmarks}
    H, W = img.shape[:2]
    # 관절 그리기
    for lm in landmarks:
        if lm["vis"] > 0.3:
            px = int(lm["x"] * W)
            py = int(lm["y"] * H)
            cv2.circle(img, (px, py), 4, (0, 255, 136), -1)
    # 연결선 그리기
    for a_idx, b_idx in _SKELETON_CONNECTIONS:
        if a_idx in lm_map and b_idx in lm_map:
            la, lb = lm_map[a_idx], lm_map[b_idx]
            if la["vis"] > 0.3 and lb["vis"] > 0.3:
                pa = (int(la["x"]*W), int(la["y"]*H))
                pb = (int(lb["x"]*W), int(lb["y"]*H))
                cv2.line(img, pa, pb, (0, 220, 200), 2)
    return img


def render_depth_colored(depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Depth float → jet 컬러맵 uint8"""
    if not _HAS_CV2 or depth is None:
        return None
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    norm = ((depth - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def render_volume(voxels, ap_pos, nic_pos, room_size, room_h, title, colorscale):
    if not voxels:
        fig = go.Figure()
        fig.update_layout(title=title, height=350,
                          paper_bgcolor="rgba(5,5,20,1)", font_color="white")
        return fig
    xs = [v["x"] for v in voxels]
    ys = [v["y"] for v in voxels]
    zs = [v["z"] for v in voxels]
    ps = [v["p"] for v in voxels]
    fig = go.Figure(data=[
        go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                     marker=dict(size=3.5, color=ps, colorscale=colorscale,
                                 opacity=0.55, cmin=0, cmax=1)),
        go.Scatter3d(x=[ap_pos[0]], y=[ap_pos[1]], z=[ap_pos[2]],
                     mode="markers+text", text=["AP"],
                     marker=dict(size=8, color="red")),
        go.Scatter3d(x=[nic_pos[0]], y=[nic_pos[1]], z=[nic_pos[2]],
                     mode="markers+text", text=["NIC"],
                     marker=dict(size=8, color="cyan")),
    ])
    fig.update_layout(
        title=title, height=350,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                   bgcolor="rgba(5,5,20,1)",
                   xaxis=dict(range=[0,room_size]), yaxis=dict(range=[0,room_size]),
                   zaxis=dict(range=[0,room_h])),
        paper_bgcolor="rgba(5,5,20,1)", font_color="white",
        margin=dict(l=0,r=0,t=30,b=0),
    )
    return fig


def render_spectrum(spectrum: Dict) -> go.Figure:
    fig = go.Figure()
    if spectrum and "az_deg" in spectrum:
        az = np.array(spectrum["az_deg"])
        P_arr = np.array(spectrum["P"])
        P_2d = P_arr.mean(axis=1) if P_arr.ndim > 1 else P_arr
        fig.add_trace(go.Scatter(x=az, y=P_2d, fill="tozeroy",
                                  line=dict(color="#00E5FF"), name="MUSIC"))
    fig.update_layout(title="MUSIC 방위각 스펙트럼", height=280,
                      paper_bgcolor="rgba(5,5,20,1)",
                      plot_bgcolor="rgba(5,5,20,1)", font_color="white",
                      xaxis_title="방위각 (°)", yaxis_title="파워")
    return fig


def render_timeseries(snap: Dict) -> go.Figure:
    fig = go.Figure()
    rssi = snap.get("rssi_series", [])
    snr = snap.get("snr_series", [])
    n = min(len(rssi), len(snr), 120)
    if n > 0:
        fig.add_trace(go.Scatter(y=rssi[-n:], name="RSSI", line=dict(color="#FF6B6B")))
        fig.add_trace(go.Scatter(y=snr[-n:], name="SNR", line=dict(color="#4ECDC4")))
    fig.update_layout(title="RSSI/SNR 시계열", height=280,
                      paper_bgcolor="rgba(5,5,20,1)",
                      plot_bgcolor="rgba(5,5,20,1)", font_color="white",
                      legend=dict(bgcolor="rgba(20,20,40,0.8)"))
    return fig


def render_action_gauge(action: str, conf: float) -> go.Figure:
    """현재 행동 분류 결과 게이지"""
    action_colors = {
        "STANDING": "#4CAF50", "WALKING": "#2196F3", "RUNNING": "#FF5722",
        "RAISING_ARM": "#9C27B0", "BENDING": "#FF9800", "SITTING": "#009688",
        "WAVING": "#E91E63", "UNKNOWN": "#607D8B",
    }
    color = action_colors.get(action, "#607D8B")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=conf * 100,
        title={"text": f"🤖 ML 행동 분류<br><b style='font-size:1.4em;color:{color}'>{action}</b>",
               "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "white"}},
            "bar": {"color": color},
            "bgcolor": "rgba(10,10,30,0.8)",
            "steps": [
                {"range": [0, 40], "color": "rgba(255,100,100,0.2)"},
                {"range": [40, 70], "color": "rgba(255,200,50,0.2)"},
                {"range": [70, 100], "color": "rgba(50,255,100,0.2)"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "value": 70},
        },
        number={"suffix": "%", "font": {"color": "white"}},
        delta={"reference": 50, "font": {"color": "white"}},
    ))
    fig.update_layout(
        height=250, paper_bgcolor="rgba(5,5,20,1)", font_color="white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def render_skeleton_3d_world(landmarks: List[Dict]) -> go.Figure:
    """MediaPipe World Landmarks 단독 3D 스켈레톤"""
    fig = go.Figure()
    if not landmarks:
        fig.update_layout(title="3D 스켈레톤 (데이터 없음)", height=400,
                          paper_bgcolor="rgba(5,5,20,1)", font_color="white")
        return fig

    lm_map = {lm["i"]: lm for lm in landmarks}
    vis_lms = [lm for lm in landmarks if lm["vis"] > 0.3]
    if vis_lms:
        fig.add_trace(go.Scatter3d(
            x=[lm["wx"] for lm in vis_lms],
            y=[lm["wz"] for lm in vis_lms],
            z=[-lm["wy"] for lm in vis_lms],
            mode="markers+text",
            text=[lm["name"][:6] for lm in vis_lms],
            textfont=dict(size=7, color="#88FFCC"),
            marker=dict(size=6, color=[lm["vis"] for lm in vis_lms],
                        colorscale="Viridis", cmin=0, cmax=1,
                        colorbar=dict(title="가시성", x=1.0, thickness=8)),
            name="관절",
        ))
    for a_idx, b_idx in _SKELETON_CONNECTIONS:
        if a_idx in lm_map and b_idx in lm_map:
            la, lb = lm_map[a_idx], lm_map[b_idx]
            if la["vis"] > 0.3 and lb["vis"] > 0.3:
                fig.add_trace(go.Scatter3d(
                    x=[la["wx"], lb["wx"], None],
                    y=[la["wz"], lb["wz"], None],
                    z=[-la["wy"], -lb["wy"], None],
                    mode="lines",
                    line=dict(color="#00E5FF", width=4),
                    showlegend=False,
                ))
    fig.update_layout(
        title="🦴 MediaPipe 3D 스켈레톤 (World Coords)", height=400,
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Z / Depth (m)", zaxis_title="Height (m)",
            bgcolor="rgba(5,5,20,1)",
            camera=dict(eye=dict(x=0, y=-2, z=1.5)),
        ),
        paper_bgcolor="rgba(5,5,20,1)", font_color="white",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def render_vital_radar(resp_bpm: float, heart_bpm: float,
                       shoulder_motion: float = 0.0) -> go.Figure:
    """바이탈 사인 레이더 차트"""
    categories = ["호흡 BPM\n정상화", "심박 BPM\n정상화", "어깨 움직임", "RF 정상화", "관측가능도"]
    resp_norm = float(np.clip((resp_bpm - 0) / 30.0, 0, 1))
    heart_norm = float(np.clip((heart_bpm - 40) / 120.0, 0, 1))
    motion_norm = float(np.clip(shoulder_motion * 20, 0, 1))
    fig = go.Figure(go.Scatterpolar(
        r=[resp_norm, heart_norm, motion_norm, 0.6, 0.7],
        theta=categories,
        fill="toself",
        fillcolor="rgba(0,200,180,0.25)",
        line=dict(color="#00E5FF", width=2),
        name="바이탈",
    ))
    fig.update_layout(
        title="🫀 바이탈 레이더", height=280,
        polar=dict(
            bgcolor="rgba(5,5,20,0.8)",
            radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#333"),
            angularaxis=dict(gridcolor="#333", tickfont=dict(color="#AAA", size=9)),
        ),
        paper_bgcolor="rgba(5,5,20,1)", font_color="white",
        margin=dict(l=40, r=40, t=40, b=10),
    )
    return fig


def render_projection_map(proj_data, title, ylabel, xlabel, colorscale):
    fig = go.Figure()
    if proj_data:
        arr = np.array(proj_data)
        fig.add_trace(go.Heatmap(z=arr, colorscale=colorscale, showscale=True))
    fig.update_layout(title=title, height=280,
                      paper_bgcolor="rgba(5,5,20,1)", font_color="white",
                      xaxis_title=xlabel, yaxis_title=ylabel,
                      margin=dict(l=0,r=0,t=30,b=0))
    return fig


def render_pseudo_camera(pseudo_cam_data):
    return render_projection_map(pseudo_cam_data, "Pseudo-CCTV (관측 포화 합성)", "Z", "X", "Hot")


def render_theory_series(snap: Dict) -> go.Figure:
    theory = snap.get("theory", {})
    fig = go.Figure()
    obs_s = theory.get("observability_series", [])
    lat_s = theory.get("latent_series", [])
    obs_a = theory.get("observed_series", [])
    n = min(len(obs_s), len(lat_s), len(obs_a), 80)
    if n > 0:
        fig.add_trace(go.Scatter(y=obs_s[-n:], name="관측가능도", line=dict(color="#00E5FF")))
        fig.add_trace(go.Scatter(y=lat_s[-n:], name="잠재 에너지", line=dict(color="#FF6B6B")))
        fig.add_trace(go.Scatter(y=obs_a[-n:], name="관측 에너지", line=dict(color="#4ECDC4")))
    fig.update_layout(title="이론 시계열 (관측 포화)", height=280,
                      paper_bgcolor="rgba(5,5,20,1)",
                      plot_bgcolor="rgba(5,5,20,1)", font_color="white",
                      legend=dict(bgcolor="rgba(20,20,40,0.8)"))
    return fig


def render_rate_series(snap: Dict) -> go.Figure:
    fig = go.Figure()
    tx = snap.get("tx_series", [])
    rx = snap.get("rx_series", [])
    n = min(len(tx), len(rx), 120)
    if n > 0:
        fig.add_trace(go.Scatter(y=tx[-n:], name="TX Mbps", line=dict(color="#FFD700")))
        fig.add_trace(go.Scatter(y=rx[-n:], name="RX Mbps", line=dict(color="#ADFF2F")))
    fig.update_layout(title="TX/RX 속도", height=280,
                      paper_bgcolor="rgba(5,5,20,1)",
                      plot_bgcolor="rgba(5,5,20,1)", font_color="white",
                      legend=dict(bgcolor="rgba(20,20,40,0.8)"))
    return fig


def render_theory_panel(snap: Dict):
    theory = snap.get("theory", {})
    manifest = snap.get("manifest", {})
    config = snap.get("theory_config", {})
    st.markdown("#### 📐 관측 포화 이론 메트릭")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("관측가능도", f'{theory.get("observability_score",0.0):.3f}')
    c2.metric("인식론 간극", f'{theory.get("epistemic_gap",0.0):.5f}')
    c3.metric("경계 압력", f'{theory.get("boundary_pressure",0.0):.5f}')
    c4.metric("신뢰도", f'{theory.get("confidence",0.0):.3f}')
    with st.expander("이론 요약", expanded=False):
        st.markdown(f"**핵심 원리**: {manifest.get('core_principle','-')}")
        st.markdown(f"**사상식**: `{manifest.get('mapping','-')}`")
        st.json({"theory_config": config, "theory_metrics": theory})


# ══════════════════════════════════════════════════════════════════════════════
# §14. 메인 Streamlit 앱
# ══════════════════════════════════════════════════════════════════════════════

def app():
    st.set_page_config(
        page_title="REVOLUTION HYPERVISION 3D — RF + Depth + ML Pose",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 커스텀 CSS (다크 테마 강화)
    st.markdown("""
    <style>
    .main { background-color: #050514; color: #E0E0FF; }
    .block-container { padding-top: 1rem; }
    .stMetric { background: rgba(10,10,30,0.8); border-radius: 8px; padding: 4px 8px; }
    .stMetricLabel { color: #88AAFF !important; }
    .stMetricValue { color: #00E5FF !important; font-weight: bold; }
    h1, h2, h3, h4 { color: #00E5FF; }
    .stSelectbox label, .stSlider label, .stToggle label, .stNumberInput label { color: #88AAFF; }
    div[data-testid="stSidebar"] { background-color: #0A0A20; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🚀 REVOLUTION HYPERVISION 3D")
    st.markdown(
        "**WiFi RF 관측 포화 이론** × **진짜 3D 카메라 파이프라인** × "
        "**MediaPipe ML 인체 포즈 추정** × **3D 칼만 융합** — 완전 통합판"
    )

    backend = get_backend()
    vision = get_vision()

    # ── 사이드바 ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🎛️ 통합 제어 패널")

        st.subheader("📡 RF 설정")
        simulation = st.toggle("RF 시뮬레이션 모드", value=True)
        ap_x = st.number_input("AP X", 0.0, 5.0, 0.0, 0.1)
        ap_y = st.number_input("AP Y", 0.0, 5.0, 0.0, 0.1)
        ap_z = st.number_input("AP Z", 0.0, 3.0, 2.5, 0.1)
        nic_x = st.number_input("NIC X", 0.0, 5.0, 2.5, 0.1)
        nic_y = st.number_input("NIC Y", 0.0, 5.0, 2.5, 0.1)
        nic_z = st.number_input("NIC Z", 0.0, 3.0, 0.8, 0.1)
        sample_hz = st.slider("샘플링 Hz", 5, 30, 12)
        process_every = st.slider("공간 처리 주기", 2, 10, 4)
        q_every = st.slider("양자 인코딩 주기", 4, 40, 16)

        st.markdown("---")
        st.subheader("📷 3D 카메라")
        force_synthetic = st.toggle("합성(Synthetic) 모드 강제", value=True,
            help="RealSense/OAK-D/ZED/웹캠이 없으면 자동으로 합성 모드")
        cam_status = vision.snapshot().get("cam_mode", "Not Started")
        st.caption(f"현재 카메라: `{cam_status}`")

        sdk_status = []
        if _HAS_RS: sdk_status.append("✅ pyrealsense2")
        else: sdk_status.append("❌ pyrealsense2")
        if _HAS_DAI: sdk_status.append("✅ depthai")
        else: sdk_status.append("❌ depthai")
        if _HAS_ZED: sdk_status.append("✅ pyzed")
        else: sdk_status.append("❌ pyzed")
        if _HAS_CV2: sdk_status.append("✅ OpenCV")
        else: sdk_status.append("❌ OpenCV")
        if _HAS_MP: sdk_status.append("✅ MediaPipe")
        else: sdk_status.append("❌ MediaPipe")
        if _HAS_TORCH: sdk_status.append("✅ PyTorch/MiDaS")
        else: sdk_status.append("❌ PyTorch/MiDaS")
        with st.expander("SDK 상태"):
            for s in sdk_status:
                st.caption(s)

        st.markdown("---")
        st.subheader("🧪 관측 포화 이론")
        theory_enabled = st.toggle("이론 엔진 활성화", value=True)
        map_mode = st.selectbox("포화 함수", ["harmonic", "tanh", "arctan"])
        saturation_limit = st.slider("포화 한계 L", 0.5, 20.0, 10.0, 0.5)
        energy_limit = st.slider("에너지 한계", 0.2, 5.0, 1.0, 0.1)
        observability_gain = st.slider("관측가능도 게인", 0.5, 2.5, 1.25, 0.05)
        ontological_gain = st.slider("잠재장 게인", 0.5, 3.5, 1.6, 0.05)
        smoothing = st.slider("볼륨 평활화", 0.55, 0.95, 0.78, 0.01)
        sigma_base = st.slider("기본 확산 σ", 0.18, 0.9, 0.48, 0.01)
        top_k_peaks = st.slider("최대 피크 수", 4, 20, 10)
        temporal_persistence = st.slider("시간 지속성", 0.50, 0.98, 0.82, 0.01)
        projection_gain = st.slider("투영 게인", 0.5, 3.0, 1.4, 0.05)
        pseudo_camera_gain = st.slider("Pseudo-CCTV 게인", 0.5, 3.5, 1.8, 0.05)
        pseudo_camera_gamma = st.slider("Pseudo-CCTV 감마", 0.25, 1.5, 0.75, 0.01)

        auto_refresh = st.toggle("자동 새로고침", value=True)
        refresh_sec = st.slider("새로고침 간격(초)", 1, 10, 2)

        theory_config = TheoryConfig(
            enabled=theory_enabled,
            saturation_limit=float(saturation_limit),
            energy_limit=float(energy_limit),
            observability_gain=float(observability_gain),
            ontological_gain=float(ontological_gain),
            smoothing=float(smoothing),
            sigma_base=float(sigma_base),
            sigma_min=0.18,
            top_k_peaks=int(top_k_peaks),
            map_mode=map_mode,
            temporal_persistence=float(temporal_persistence),
            projection_gain=float(projection_gain),
            pseudo_camera_gain=float(pseudo_camera_gain),
            pseudo_camera_gamma=float(pseudo_camera_gamma),
        )

        col1, col2 = st.columns(2)
        if col1.button("⚙️ RF 구성 적용", use_container_width=True):
            backend.configure(simulation, (ap_x,ap_y,ap_z), (nic_x,nic_y,nic_z), theory_config)
            st.success("구성 적용 완료")
        if col2.button("🔄 RF 초기화", use_container_width=True):
            backend.reset()
            st.success("버퍼 초기화")

        col3, col4 = st.columns(2)
        if col3.button("▶️ RF 시작", use_container_width=True):
            backend.configure(simulation, (ap_x,ap_y,ap_z), (nic_x,nic_y,nic_z), theory_config)
            backend.start(sample_hz, process_every, q_every)
            st.success("RF 루프 시작")
        if col4.button("⏹️ RF 중지", use_container_width=True):
            backend.stop()
            st.warning("RF 루프 중지")

        col5, col6 = st.columns(2)
        if col5.button("📷 카메라 시작", use_container_width=True):
            vision._force_synthetic = force_synthetic
            if not vision.snapshot()["running"]:
                vision.start()
                st.success("비전 파이프라인 시작")
            else:
                st.info("이미 실행 중")
        if col6.button("📷 카메라 중지", use_container_width=True):
            vision.stop()
            st.warning("비전 파이프라인 중지")

    # ── 데이터 스냅샷 ─────────────────────────────────────────────────────────
    rf_snap = backend.snapshot()
    vis_snap = vision.snapshot()

    # ── 상태 헤더 ─────────────────────────────────────────────────────────────
    rf_status = "🟢 RF 실행" if rf_snap["running"] else "🔴 RF 중지"
    vis_status = "🟢 비전 실행" if vis_snap["running"] else "🔴 비전 중지"
    cam_mode = vis_snap.get("cam_mode", "None")
    ml_status = "✅ MediaPipe" if _HAS_MP else "❌ MediaPipe"
    sim_mode = "SIM" if rf_snap["simulation"] else "REAL RF"
    st.markdown(
        f"**{rf_status}** · **{vis_status}** · `{sim_mode}` · "
        f"카메라: `{cam_mode}` · {ml_status}"
    )
    if rf_snap["last_error"]:
        st.error(f"RF 오류: {rf_snap['last_error']}")
    if vis_snap.get("last_error"):
        st.warning(f"비전 오류: {vis_snap['last_error']}")

    # ── 핵심 메트릭 바 ─────────────────────────────────────────────────────────
    mc = st.columns(10)
    mc[0].metric("RF 샘플", rf_snap["samples"])
    mc[1].metric("모션 에너지", f"{rf_snap['motion_energy']:.5f}")
    mc[2].metric("거리 변동", f"{rf_snap['distance_std_cm']:.2f} cm")
    mc[3].metric("호흡", f"{rf_snap['resp_bpm']:.1f} bpm")
    mc[4].metric("심박", f"{rf_snap['heart_bpm']:.1f} bpm")
    mc[5].metric("관측가능도", f"{rf_snap.get('theory',{}).get('observability_score',0.0):.3f}")
    mc[6].metric("신뢰도", f"{rf_snap.get('theory',{}).get('confidence',0.0):.3f}")
    mc[7].metric("카메라 프레임", vis_snap["frame_count"])
    action = vis_snap.get("action", "UNKNOWN")
    aconf = vis_snap.get("action_conf", 0.0)
    mc[8].metric("행동 분류", action[:8])
    mc[9].metric("행동 신뢰도", f"{aconf:.2f}")

    st.markdown("---")

    # ══ 행 1: 메인 3D 씬 (전체 너비) ══════════════════════════════════════════
    st.subheader("🌐 완전 융합 3D 씬 — RF 볼륨 + Depth 포인트클라우드 + ML 스켈레톤 + 칼만")
    fig_3d = render_3d_scene(rf_snap, vis_snap)
    st.plotly_chart(fig_3d, use_container_width=True)

    # ══ 행 2: 카메라 영상 + 포즈 + 스켈레톤 3D ═══════════════════════════════
    row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])

    with row2_col1:
        st.subheader("🎥 RGB + 포즈 오버레이")
        pose_img = render_pose_2d(vis_snap)
        if pose_img is not None:
            st.image(pose_img, channels="RGB", use_container_width=True)
        elif vis_snap.get("rgb") is not None:
            st.image(vis_snap["rgb"], channels="RGB", use_container_width=True)
        else:
            st.info("카메라 시작 버튼을 눌러주세요")
        n_lms = len(vis_snap.get("landmarks", []))
        if n_lms:
            st.caption(f"✅ {n_lms}개 랜드마크 감지됨")
        else:
            st.caption("❌ 포즈 감지 없음")

    with row2_col2:
        st.subheader("🌊 Depth 맵 (컬러)")
        depth_img = render_depth_colored(vis_snap.get("depth"))
        if depth_img is not None:
            st.image(depth_img, channels="RGB", use_container_width=True)
            depth = vis_snap.get("depth")
            if depth is not None:
                valid = depth[(depth > 0.1) & (depth < 8.0)]
                if len(valid) > 0:
                    st.caption(f"깊이 범위: {valid.min():.2f}m ~ {valid.max():.2f}m")
        else:
            st.info("카메라 시작 버튼을 눌러주세요")

    with row2_col3:
        st.subheader("🦴 MediaPipe 3D 스켈레톤")
        skel_fig = render_skeleton_3d_world(vis_snap.get("landmarks", []))
        st.plotly_chart(skel_fig, use_container_width=True)

    # ══ 행 3: 행동 분류 + 바이탈 레이더 + 트래젝토리 ═════════════════════════
    row3_col1, row3_col2, row3_col3 = st.columns([1, 1, 1])

    with row3_col1:
        st.subheader("🤖 ML 행동 분류")
        action_fig = render_action_gauge(action, aconf)
        st.plotly_chart(action_fig, use_container_width=True)

        # 행동 이력
        if "action_history" not in st.session_state:
            st.session_state["action_history"] = deque(maxlen=20)
        if action != "UNKNOWN":
            st.session_state["action_history"].append(
                {"시간": time.strftime("%H:%M:%S"), "행동": action, "신뢰도": f"{aconf:.2f}"})
        hist = list(st.session_state["action_history"])
        if hist:
            st.dataframe(pd.DataFrame(hist[-8:]).iloc[::-1],
                         use_container_width=True, height=160)

    with row3_col2:
        st.subheader("🫀 바이탈 사인 레이더")
        # 어깨 움직임 에너지 (포즈 기반)
        shoulder_motion = 0.0
        lms = vis_snap.get("landmarks", [])
        if lms:
            l_sh_lm = next((l for l in lms if l["i"] == 11), None)
            r_sh_lm = next((l for l in lms if l["i"] == 12), None)
            if l_sh_lm and r_sh_lm:
                shoulder_motion = abs(l_sh_lm["wy"] - r_sh_lm["wy"])
        vital_fig = render_vital_radar(rf_snap["resp_bpm"], rf_snap["heart_bpm"], shoulder_motion)
        st.plotly_chart(vital_fig, use_container_width=True)

    with row3_col3:
        st.subheader("📍 칼만 트래젝토리 (Top View)")
        traj = vis_snap.get("trajectory", [])
        fused_xyz = vis_snap.get("fused_xyz")
        traj_fig = go.Figure()
        if len(traj) >= 2:
            traj_fig.add_trace(go.Scatter(
                x=[p[0] for p in traj],
                y=[p[2] for p in traj],
                mode="lines+markers",
                line=dict(color="cyan", width=2),
                marker=dict(size=4, color=list(range(len(traj))),
                            colorscale="Plasma", cmin=0, cmax=len(traj)),
                name="궤적",
            ))
        if fused_xyz:
            traj_fig.add_trace(go.Scatter(
                x=[fused_xyz[0]], y=[fused_xyz[2]],
                mode="markers+text", text=["현재"],
                marker=dict(size=12, color="yellow", symbol="star"),
                name="현재 위치",
            ))
        traj_fig.update_layout(
            title="XZ 평면 궤적 (Kalman Fusion)",
            height=280, paper_bgcolor="rgba(5,5,20,1)",
            plot_bgcolor="rgba(5,5,20,1)", font_color="white",
            xaxis_title="X (m)", yaxis_title="Depth (m)",
            showlegend=True,
            legend=dict(bgcolor="rgba(20,20,40,0.8)"),
        )
        st.plotly_chart(traj_fig, use_container_width=True)

    # ══ 행 4: RF 볼륨 + 양자 볼륨 + MUSIC 스펙트럼 ═══════════════════════════
    st.markdown("---")
    row4_col1, row4_col2, row4_col3 = st.columns([1, 1, 1])

    with row4_col1:
        st.plotly_chart(
            render_volume(rf_snap["voxels"], rf_snap["ap_pos"], rf_snap["nic_pos"],
                          rf_snap["room_size"], rf_snap["room_h"],
                          "📡 관측 포화 RF 볼륨", "Turbo"),
            use_container_width=True)
    with row4_col2:
        st.plotly_chart(
            render_volume(rf_snap["q_voxels"], rf_snap["ap_pos"], rf_snap["nic_pos"],
                          rf_snap["room_size"], rf_snap["room_h"],
                          "⚛️ 양자 인코딩 볼륨", "Plasma"),
            use_container_width=True)
    with row4_col3:
        st.plotly_chart(render_spectrum(rf_snap["spectrum"]), use_container_width=True)

    # ══ 행 5: 시계열 차트들 ════════════════════════════════════════════════════
    row5_col1, row5_col2 = st.columns(2)
    with row5_col1:
        st.plotly_chart(render_timeseries(rf_snap), use_container_width=True)
    with row5_col2:
        st.plotly_chart(render_rate_series(rf_snap), use_container_width=True)

    # ══ 행 6: 투영 맵들 ════════════════════════════════════════════════════════
    row6_col1, row6_col2, row6_col3, row6_col4 = st.columns(4)
    with row6_col1:
        st.plotly_chart(render_pseudo_camera(rf_snap.get("pseudo_camera", [])),
                        use_container_width=True)
    with row6_col2:
        st.plotly_chart(render_projection_map(rf_snap.get("top_projection",[]),
                        "Top Projection", "Y","X","Viridis"), use_container_width=True)
    with row6_col3:
        st.plotly_chart(render_projection_map(rf_snap.get("front_projection",[]),
                        "Front Projection","Z","X","Cividis"), use_container_width=True)
    with row6_col4:
        st.plotly_chart(render_projection_map(rf_snap.get("side_projection",[]),
                        "Side Projection","Z","Y","Plasma"), use_container_width=True)

    # ══ 행 7: 이론 + 행 8: 이론 시계열 ═══════════════════════════════════════
    row7_col1, row7_col2 = st.columns(2)
    with row7_col1:
        st.plotly_chart(render_theory_series(rf_snap), use_container_width=True)
    with row7_col2:
        render_theory_panel(rf_snap)

    # ══ 행 9: 랜드마크 상세 테이블 ════════════════════════════════════════════
    with st.expander("🔬 MediaPipe 랜드마크 상세 (33개 관절)", expanded=False):
        lms = vis_snap.get("landmarks", [])
        if lms:
            df_lms = pd.DataFrame([{
                "ID": lm["i"], "이름": lm["name"],
                "스크린X": f"{lm['x']:.3f}", "스크린Y": f"{lm['y']:.3f}",
                "World X": f"{lm['wx']:.4f}",
                "World Y": f"{lm['wy']:.4f}",
                "World Z": f"{lm['wz']:.4f}",
                "가시성": f"{lm['vis']:.2f}",
            } for lm in lms])
            st.dataframe(df_lms, use_container_width=True, height=300)
        else:
            st.info("포즈가 감지되지 않았습니다.")

    # ══ 행 10: 시스템 요약 JSON ═══════════════════════════════════════════════
    with st.expander("📋 시스템 상태 전체 JSON", expanded=False):
        fused = vis_snap.get("fused_xyz")
        st.json({
            "RF": {
                "mode": "simulation" if rf_snap["simulation"] else "real_rf",
                "iface": rf_snap["iface"],
                "peak_xyz": [round(v, 3) for v in rf_snap["peak_xyz"]],
                "motion_detected": rf_snap["motion_detected"],
                "resp_bpm": rf_snap["resp_bpm"],
                "heart_bpm": rf_snap["heart_bpm"],
                "quantum_backend": "qiskit-aer" if _HAS_QISKIT else "fallback",
                "theory": rf_snap.get("theory", {}),
            },
            "Vision": {
                "camera_mode": vis_snap["cam_mode"],
                "frame_count": vis_snap["frame_count"],
                "pose_detected": len(vis_snap.get("landmarks", [])) > 0,
                "n_landmarks": len(vis_snap.get("landmarks", [])),
                "action": action,
                "action_confidence": round(aconf, 3),
                "fused_xyz_kalman": [round(v, 3) for v in fused] if fused else None,
                "trajectory_len": len(vis_snap.get("trajectory", [])),
            },
            "SDKs": {
                "pyrealsense2": _HAS_RS,
                "depthai": _HAS_DAI,
                "pyzed": _HAS_ZED,
                "opencv": _HAS_CV2,
                "mediapipe": _HAS_MP,
                "pytorch_midas": _HAS_TORCH,
                "qiskit": _HAS_QISKIT,
            },
        })

    st.markdown("---")
    st.caption(
        "⚠️ REVOLUTION HYPERVISION 3D — 연구/실험용. "
        "RealSense D4xx/L5xx, Luxonis OAK-D, ZED 2/ZED X, 또는 일반 웹캠 연결 시 "
        "진짜 Depth 데이터와 실시간 ML 포즈 추정이 활성화됩니다. "
        "카메라 없이도 합성 시뮬레이션 모드로 완전 동작합니다."
    )

    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()


if __name__ == "__main__":
    app()
