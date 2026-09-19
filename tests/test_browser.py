"""Finding Chrome / Edge and opening the app window (no real browser is started)."""

import subprocess
import sys

import pytest

from line2func import browser


def test_find_prefers_chrome_then_edge(monkeypatch, tmp_path):
    chrome, edge = tmp_path / "chrome.exe", tmp_path / "msedge.exe"
    chrome.write_text("")
    edge.write_text("")
    paths = {"chrome": [str(tmp_path / "missing.exe"), str(chrome)], "edge": [str(edge)]}
    monkeypatch.setattr(browser, "candidates", lambda name: paths[name])
    assert browser.find() == ("chrome", str(chrome))
    assert browser.find("edge") == ("edge", str(edge))
    chrome.unlink()
    assert browser.find() == ("edge", str(edge))
    edge.unlink()
    assert browser.find() is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows search paths")
def test_windows_search_order(monkeypatch):
    monkeypatch.setattr(browser, "_registry_paths", lambda exe: [rf"C:\Registry\{exe}"])
    monkeypatch.setenv("PROGRAMFILES", r"C:\PF")
    monkeypatch.setenv("PROGRAMFILES(X86)", r"C:\PF86")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Local")
    monkeypatch.setattr(browser.shutil, "which", lambda name: rf"C:\OnPath\{name}")
    assert browser.candidates("edge") == [
        r"C:\Registry\msedge.exe",
        r"C:\PF\Microsoft\Edge\Application\msedge.exe",
        r"C:\PF86\Microsoft\Edge\Application\msedge.exe",
        r"C:\Local\Microsoft\Edge\Application\msedge.exe",
        r"C:\OnPath\msedge.exe",
    ]


def test_launch_opens_a_detached_app_window(monkeypatch):
    calls = []
    monkeypatch.setattr(browser, "find", lambda pref="auto": ("edge", "C:/edge.exe"))
    monkeypatch.setattr(browser.subprocess, "Popen", lambda args, **kw: calls.append((args, kw)))
    assert browser.launch("http://127.0.0.1:1234/") == "edge"
    args, kw = calls[0]
    assert args == ["C:/edge.exe", "--app=http://127.0.0.1:1234/"]
    assert kw["stdout"] is subprocess.DEVNULL and kw["stdin"] is subprocess.DEVNULL
    if sys.platform == "win32":
        flags = kw["creationflags"]
        assert flags & subprocess.DETACHED_PROCESS and flags & subprocess.CREATE_BREAKAWAY_FROM_JOB
    else:
        assert kw["start_new_session"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
def test_launch_retries_when_breaking_away_is_not_allowed(monkeypatch):
    flags = []

    def popen(args, **kw):
        flags.append(kw["creationflags"])
        if kw["creationflags"] & subprocess.CREATE_BREAKAWAY_FROM_JOB:
            raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(browser, "find", lambda pref="auto": ("chrome", "C:/chrome.exe"))
    monkeypatch.setattr(browser.subprocess, "Popen", popen)
    assert browser.launch("http://127.0.0.1:1/") == "chrome"
    assert len(flags) == 2 and not flags[1] & subprocess.CREATE_BREAKAWAY_FROM_JOB


def test_launch_falls_back_to_the_default_browser(monkeypatch):
    opened = []
    monkeypatch.setattr(browser, "find", lambda pref="auto": None)
    monkeypatch.setattr(browser.webbrowser, "open", lambda url: opened.append(url) or True)
    assert browser.launch("http://x/") == "default" and opened == ["http://x/"]
    assert browser.launch("http://x/", "default") == "default"
    assert browser.launch("http://x/", "none") == "none" and len(opened) == 2
    with pytest.raises(ValueError):
        browser.launch("http://x/", "firefox")
