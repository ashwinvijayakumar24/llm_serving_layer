"""
Hardware capture for the artifact's `hardware` block, plus an optional
low-rate GPU clock/temperature sampler.

Every field is best-effort and `None` when it cannot be read — never a guess.
On a laptop or a CPU-only CI runner `gpu_name` is None, and `publishable()`
says so: a cross-engine number without the GPU it ran on cannot be compared
with anything (SPEC hard rule 5, R12).

Sources, in order of preference:
  gpu_name / gpu_count  nvidia-smi, then torch.cuda
  driver                nvidia-smi --query-gpu=driver_version
  cuda                  the driver's CUDA version from the nvidia-smi banner,
                        then torch.version.cuda (the toolkit torch was built
                        with — a different number, so the source is recorded)
  node                  SLURMD_NODENAME, then the hostname
  slurm_job_id          SLURM_JOB_ID
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import threading
import time
from typing import Any


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def parse_nvidia_smi_query(text: str | None) -> list[dict[str, str]]:
    """`--query-gpu=name,driver_version --format=csv,noheader` -> rows."""
    rows = []
    for line in (text or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            rows.append({"name": parts[0], "driver": parts[1]})
    return rows


def parse_cuda_banner(text: str | None) -> str | None:
    m = re.search(r"CUDA Version:\s*([0-9.]+)", text or "")
    return m.group(1) if m else None


def capture_hardware() -> dict[str, Any]:
    hw: dict[str, Any] = {
        "gpu_name": None,
        "gpu_count": None,
        "driver": None,
        "cuda": None,
        "cuda_source": None,
        "torch_cuda": None,
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    rows = parse_nvidia_smi_query(
        _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    )
    if rows:
        hw["gpu_name"] = rows[0]["name"]
        hw["gpu_count"] = len(rows)
        hw["driver"] = rows[0]["driver"]
    banner = parse_cuda_banner(_run(["nvidia-smi"]))
    if banner:
        hw["cuda"], hw["cuda_source"] = banner, "nvidia-smi (driver CUDA version)"
    try:
        import torch

        hw["torch_cuda"] = torch.version.cuda
        if hw["cuda"] is None and torch.version.cuda:
            hw["cuda"], hw["cuda_source"] = torch.version.cuda, "torch.version.cuda"
        if hw["gpu_name"] is None and torch.cuda.is_available():
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["gpu_count"] = torch.cuda.device_count()
    except Exception:  # noqa: BLE001 — absent torch is a None, not an error
        pass
    return hw


def publishable(hw: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if not hw.get("gpu_name"):
        reasons.append("no GPU detected: a cross-engine number needs the GPU it ran on")
    if not hw.get("slurm_job_id"):
        reasons.append("no SLURM_JOB_ID: allocation identity unknown (SPEC rule 5, R12)")
    return (not reasons, reasons)


class GpuSampler:
    """
    nvidia-smi poller on a background thread, default every 2 s. Low rate on
    purpose: it shares the host with the load generator, and a sampler that
    steals CPU from the client shows up as dispatch drift. Does nothing (and
    records nothing) when nvidia-smi is absent.
    """

    QUERY = "clocks.sm,temperature.gpu,power.draw"

    def __init__(self, interval_s: float = 2.0, gpu_index: int = 0) -> None:
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = shutil.which("nvidia-smi") is not None

    @staticmethod
    def parse(line: str) -> dict[str, Any] | None:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 3:
            return None

        def num(x: str) -> float | None:
            try:
                return float(x)
            except ValueError:
                return None

        return {"sm_clock_mhz": num(parts[0]), "temp_c": num(parts[1]), "power_w": num(parts[2])}

    def _loop(self) -> None:
        t0 = time.monotonic()
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            f"--query-gpu={self.QUERY}",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            out = _run(cmd, timeout=5.0)
            if out:
                s = self.parse(out.splitlines()[0])
                if s is not None:
                    s["t"] = time.monotonic() - t0
                    self.samples.append(s)
            self._stop.wait(self.interval_s)

    def start(self) -> GpuSampler:
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return list(self.samples)
