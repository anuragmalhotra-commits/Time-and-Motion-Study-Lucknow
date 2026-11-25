@echo off
cd /d "%~dp0"
set "PYTHONPATH=C:\repnetvenv\Lib\site-packages;C:\repnet_pkgs;%PYTHONPATH%"
if exist ".venv\Scripts\python.exe" (
    set "VPY=.venv\Scripts\python.exe"
) else if exist "C:\repnetvenv\Scripts\python.exe" (
    set "VPY=C:\repnetvenv\Scripts\python.exe"
) else (
    set "VPY=python"
)
"%VPY%" self_test.py
pause
