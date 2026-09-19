"""Find Chrome or Edge and open a page in an app window (no tabs, no address bar).

Used by ``python -m line2func``. Search order on Windows: the registry's
"App Paths" (per user, then per machine), the usual install folders, then
``PATH``. On macOS and Linux the usual application paths and command names.
If neither browser is found, the system's default browser opens a normal tab.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser

BROWSERS = ("chrome", "edge")
CHOICES = ("auto", "chrome", "edge", "default", "none")

_WINDOWS = {
    "chrome": ("chrome.exe", r"Google\Chrome\Application\chrome.exe"),
    "edge": ("msedge.exe", r"Microsoft\Edge\Application\msedge.exe"),
}
_MAC = {
    "chrome": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "edge": "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
}
_UNIX = {
    "chrome": ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"),
    "edge": ("microsoft-edge", "microsoft-edge-stable"),
}


def _registry_paths(exe: str) -> list[str]:
    """``App Paths`` registry entries for ``exe`` (current user first)."""
    try:
        import winreg
    except ImportError:
        return []
    found = []
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}") as key:
                value, _ = winreg.QueryValueEx(key, "")
        except OSError:
            continue
        if value:
            found.append(str(value).strip('"'))
    return found


def candidates(name: str) -> list[str]:
    """Possible executable paths for ``name`` ("chrome" or "edge"), most likely first."""
    if sys.platform == "win32":
        exe, rel = _WINDOWS[name]
        paths = _registry_paths(exe)
        for var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            base = os.environ.get(var)
            if base:
                paths.append(os.path.join(base, rel))
        on_path = shutil.which(exe)
        if on_path:
            paths.append(on_path)
        return paths
    if sys.platform == "darwin":
        return [_MAC[name]]
    return [p for p in (shutil.which(cmd) for cmd in _UNIX[name]) if p]


def find(pref: str = "auto") -> tuple[str, str] | None:
    """``(name, executable)`` of the first browser found: Chrome, then Edge (or only ``pref``)."""
    for name in (BROWSERS if pref == "auto" else (pref,)):
        for path in candidates(name):
            if os.path.isfile(path):
                return name, path
    return None


def app_args(exe: str, url: str) -> list[str]:
    return [exe, f"--app={url}"]


def _spawn(args: list[str]) -> bool:
    """Start the browser fully detached, so closing our terminal never closes it."""
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
              "close_fds": True}
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        # leave our job object too (IDE terminals kill their job's processes on close);
        # a job that forbids breaking away makes this fail, so retry without it
        for extra in (subprocess.CREATE_BREAKAWAY_FROM_JOB, 0):
            try:
                subprocess.Popen(args, creationflags=flags | extra, **kwargs)
                return True
            except OSError:
                continue
        return False
    try:
        subprocess.Popen(args, start_new_session=True, **kwargs)
        return True
    except OSError:
        return False


def launch(url: str, pref: str = "auto") -> str:
    """Open ``url``; returns ``"chrome"`` or ``"edge"`` (an app window), ``"default"`` (a tab) or ``"none"``."""
    if pref not in CHOICES:
        raise ValueError(f"unknown browser {pref!r}; choose from {', '.join(CHOICES)}")
    if pref == "none":
        return "none"
    if pref != "default":
        found = find(pref)
        if found and _spawn(app_args(found[1], url)):
            return found[0]
    return "default" if webbrowser.open(url) else "none"
