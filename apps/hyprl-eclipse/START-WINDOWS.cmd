@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>&1
if not errorlevel 1 goto python_launcher
where python >nul 2>&1
if not errorlevel 1 goto python_command
echo Python 3 est necessaire pour ouvrir cet apercu.
echo Consultez README.md pour la commande de lancement.
pause
exit /b 1
:python_launcher
start "HYPRL - serveur local" cmd /k py -3 -m http.server 8093 --bind 127.0.0.1
goto open_browser
:python_command
start "HYPRL - serveur local" cmd /k python -m http.server 8093 --bind 127.0.0.1
:open_browser
timeout /t 2 /nobreak >nul
start "" "http://localhost:8093/"
