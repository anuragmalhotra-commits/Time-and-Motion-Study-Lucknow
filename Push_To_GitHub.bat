@echo off
setlocal
cd /d "%~dp0"
set "PATH=C:\mingit\cmd;%PATH%"

echo ======================================================================
echo   InSight Time ^& Motion Bench - Push to GitHub
echo   Target: https://github.com/anuragmalhotra-commits/Time-and-Motion-Study-Lucknow.git
echo ======================================================================
echo.

git status
echo.
echo Pushing branch 'main' to GitHub origin...
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ======================================================================
    echo   Push Succeeded! 
    echo   View repository at:
    echo   https://github.com/anuragmalhotra-commits/Time-and-Motion-Study-Lucknow
    echo ======================================================================
) else (
    echo.
    echo ----------------------------------------------------------------------
    echo   If authentication failed or prompted:
    echo   GitHub requires a Personal Access Token (PAT) instead of your password.
    echo   1. Visit: https://github.com/settings/tokens
    echo   2. Generate a token (Classic with 'repo' scope)
    echo   3. When prompted, enter your GitHub username and paste the PAT as password.
    echo ----------------------------------------------------------------------
)
echo.
pause
