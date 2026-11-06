"""OS helpers."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Callable, Iterator


def _flush_c_stdio() -> None:
    import ctypes

    try:
        if sys.platform == "win32":
            ctypes.cdll.ucrtbase.fflush(None)
        else:
            ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


@contextmanager
def silence_native_output() -> Iterator[None]:
    """Drop text that C libraries print straight to the console (stdout and stderr).

    The eSpeak library inside the Kokoro stack prints phonemes for unknown words
    (e.g. "pədkˈæstᵻfˌa‍ɪ") to stderr. Python's redirect_stdout/stderr can't catch C
    output, so we point file descriptors 1 and 2 at the null device during the call.
    Errors are still reported: they are raised as Python exceptions after this exits.
    """
    fds: list[int] = []
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
            fds.append(stream.fileno())
        except (AttributeError, OSError, ValueError):  # no real console (e.g. pythonw)
            pass
    if not fds:
        yield
        return
    saved = [(fd, os.dup(fd)) for fd in fds]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        for fd in fds:
            os.dup2(devnull, fd)
        yield
    finally:
        _flush_c_stdio()  # push buffered C output into the null device, not the console
        for fd, copy in saved:
            os.dup2(copy, fd)
            os.close(copy)
        os.close(devnull)

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


@contextmanager
def keep_awake(enabled: bool = True) -> Iterator[None]:
    """Stop Windows from sleeping (and the screen from turning off) while we work.

    Generation takes many minutes on CPU; if the laptop drops into Modern Standby,
    Ollama and TTS freeze until it wakes. Closing the lid still sleeps the machine.
    The request belongs to the calling thread, so use this around the actual work.
    """
    if not enabled or sys.platform != "win32":
        yield
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    ok = kernel32.SetThreadExecutionState(
        _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
    )
    try:
        yield
    finally:
        if ok:
            kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


_GPU_PROVIDERS = {"cuda", "gpu", "tensorrt"}
_gpu_state: bool | None = None  # None = not checked yet


def is_gpu_provider(provider: str) -> bool:
    return provider.lower() in _GPU_PROVIDERS


def prepare_gpu(provider: str, log: Callable[[str], None] = print) -> bool:
    """Get NVIDIA CUDA ready for onnxruntime (Kokoro and speech-to-text).

    Returns True when CUDA can be used. After the GPU switch in the README, CUDA and
    cuDNN come as pip packages (nvidia-*); on Windows their DLLs are not on the search
    path, so onnxruntime needs preload_dlls() before the first CUDA session.
    Returns False (warning once) when the GPU build of onnxruntime isn't installed,
    so callers can fall back to the CPU. Always False for "cpu".
    """
    global _gpu_state
    if not is_gpu_provider(provider):
        return False
    if _gpu_state is not None:
        return _gpu_state
    _gpu_state = False
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            log(
                f"  Warning: provider {provider!r} needs the GPU build of onnxruntime "
                "(see 'Switching to an NVIDIA GPU' in README.md). Using the CPU for now."
            )
            return False
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        _gpu_state = True
    except Exception as exc:  # never block a run: fall back to the CPU
        log(f"  Warning: could not load the CUDA libraries ({exc}). Using the CPU for now.")
    return _gpu_state
