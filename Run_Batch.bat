@echo off
setlocal EnableExtensions
title Batch analysis - InSight Time and Motion Bench
cd /d "%~dp0"
set "PYTHONPATH=C:\repnetvenv\Lib\site-packages;C:\repnet_pkgs;%PYTHONPATH%"
if exist ".venv\Scripts\python.exe" (
    set "VPY=.venv\Scripts\python.exe"
) else if exist "C:\repnetvenv\Scripts\python.exe" (
    set "VPY=C:\repnetvenv\Scripts\python.exe"
) else (
    set "VPY=python"
)
rem Default: adaptive learner, optical flow, hand landmarks
"%VPY%" batch.py --engines adaptive_learner,optical_flow,hand_landmarks
echo.
pause
