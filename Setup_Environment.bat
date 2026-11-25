@echo off
setlocal EnableExtensions
title Set-up - InSight Time and Motion Bench
cd /d "%~dp0"

rem ===========================================================================
rem  One-time set-up (needs internet, about 2-3 GB of downloads):
rem   1. a private Python environment in .venv with the packages it needs
rem   2. model files from their authors' pages: MediaPipe hand model, RepNet
rem      (PyTorch) code and weights, TransRAC code and Video Swin weights
rem   3. a self-test
rem  The TransRAC checkpoint itself must be downloaded by hand (see README).
rem  Safe to run again: anything already present is kept.
rem ===========================================================================

echo ===============================================================================
echo            THE/NUDGE INSTITUTE - InSight x SIDBI Chikankari Study
echo                  Setting up the Time and Motion Bench
echo ===============================================================================
echo.

rem ---- 1. Python and packages --------------------------------------------------
set "PY="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY goto :err_python
for /f "delims=" %%v in ('%PY% --version 2^>^&1') do set "PYVER=%%v"
echo [OK] Using %PYVER%

if exist ".venv\Scripts\python.exe" goto :venv_ok
echo Creating the private environment (.venv)...
%PY% -m venv .venv
if errorlevel 1 goto :err_venv
:venv_ok
set "VPY=.venv\Scripts\python.exe"

echo Installing packages - this can take 5-15 minutes...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :err_install
echo [OK] Packages installed
echo.

rem ---- 2. Model files -------------------------------------------------------------
echo Downloading model files...
"%VPY%" setup_models.py
echo.

rem ---- 3. Self-test ---------------------------------------------------------------
echo Running the self-test...
"%VPY%" self_test.py
echo.
echo [DONE] Start the app with Launch_App.bat
echo.
pause
exit /b 0

:err_python
echo [ERROR] Python 3.10, 3.11 or 3.12 was not found.
echo         Install Python 3.12 from https://www.python.org/downloads/
echo         and tick "Add python.exe to PATH", then run this file again.
goto :fail
:err_venv
echo [ERROR] Could not create the .venv folder - see the message above.
goto :fail
:err_install
echo [ERROR] Package installation failed - check the internet connection and run this again.
goto :fail
:fail
echo.
pause
exit /b 1
