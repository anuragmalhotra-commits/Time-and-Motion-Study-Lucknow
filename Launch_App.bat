@echo off
setlocal EnableExtensions
title InSight Time and Motion Bench - The/Nudge Institute
cd /d "%~dp0"
set "PORT=8501"
set "MAX_PORT=8520"

echo ===============================================================================
echo            THE/NUDGE INSTITUTE - InSight x SIDBI Chikankari Study
echo                        Time and Motion Bench
echo ===============================================================================
echo.
if not exist "app.py" goto :err_noapp
set "PYTHONPATH=C:\repnetvenv\Lib\site-packages;C:\repnet_pkgs;%PYTHONPATH%"
if exist ".venv\Scripts\python.exe" (
    set "VPY=.venv\Scripts\python.exe"
) else if exist "C:\repnetvenv\Scripts\python.exe" (
    set "VPY=C:\repnetvenv\Scripts\python.exe"
) else (
    set "VPY=python"
)

echo "%CD%" | find /i "OneDrive" >nul
if not errorlevel 1 (
    echo [WARNING] This folder is inside OneDrive. Syncing large videos can slow the app.
    echo           A local folder such as C:\InSight_TimeMotion is safer.
    echo.
)

:find_port
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if errorlevel 1 goto :port_ok
set /a PORT+=1
if %PORT% GTR %MAX_PORT% goto :err_noport
goto :find_port
:port_ok

echo Starting the app at http://localhost:%PORT%
echo Keep this window open while you use the app. Close it to stop the app.
echo.
"%VPY%" -m streamlit run app.py --global.developmentMode=false --server.port=%PORT% --server.address=localhost --browser.gatherUsageStats=false --server.fileWatcherType=none
if errorlevel 1 goto :err_crash
goto :end

:err_noapp
echo [ERROR] app.py not found. Keep this file in the app folder.
goto :fail
:err_nosetup
echo [ERROR] The app is not set up yet. Run Setup_Environment.bat first.
goto :fail
:err_noport
echo [ERROR] No free port between 8501 and %MAX_PORT%. Close other copies of the app.
goto :fail
:err_crash
echo.
echo [ERROR] The app stopped unexpectedly - please take a screenshot of this window.
goto :fail
:fail
echo.
pause
exit /b 1
:end
pause
exit /b 0
