@echo off
REM HyprL launcher helper for Windows PowerShell/CMD environments.
setlocal EnableDelayedExpansion

set "ROOT=%~dp0.."
cd /d "%ROOT%"

if not exist ".venv\Scripts\activate.bat" (
    echo Environnement virtuel (.venv) introuvable. Creez-le avant d'utiliser ce launcher :
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install --upgrade pip
    echo   pip install -r requirements.txt
    echo   pip install -e .
    exit /b 1
)

call ".venv\Scripts\activate.bat"

:menu
echo(
echo =======================
echo        HyprL Menu
echo =======================
echo 1^) Lancer GUI principal (Streamlit)
echo 2^) Lancer GUI Replay
echo 3^) Run Analysis (CLI)
echo 4^) Run Backtest (CLI)
echo 5^) Lancer pytest -q
echo 6^) Commande personnalisée
echo 0^) Quitter
set /p choice=Choix^> 

if "%choice%"=="1" goto gui
if "%choice%"=="2" goto replay
if "%choice%"=="3" goto analysis
if "%choice%"=="4" goto backtest
if "%choice%"=="5" goto tests
if "%choice%"=="6" goto custom
if "%choice%"=="0" goto end
echo Choix invalide.
goto menu

:gui
streamlit run scripts/hyprl_gui.py
goto menu

:replay
streamlit run scripts/hyprl_replay_gui.py
goto menu

:analysis
set "ticker=AAPL"
set "period=5d"
set "interval=1h"
set "model=logistic"
set "calibration=none"

set /p input=Ticker [!ticker!]^> 
if not "!input!"=="" set "ticker=!input!"
set /p input=Periode (ex: 5d, 1y) [!period!]^> 
if not "!input!"=="" set "period=!input!"
set /p input=Intervalle [!interval!]^> 
if not "!input!"=="" set "interval=!input!"
set /p input=Model (logistic/random_forest/xgboost) [!model!]^> 
if not "!input!"=="" set "model=!input!"
set /p input=Calibration (none/platt/isotonic) [!calibration!]^> 
if not "!input!"=="" set "calibration=!input!"

set "cmd=python scripts/run_analysis.py --ticker !ticker! --period !period!"
if not "!interval!"=="" set "cmd=!cmd! --interval !interval!"
if /I not "!model!"=="logistic" set "cmd=!cmd! --model-type !model!"
if /I not "!calibration!"=="none" set "cmd=!cmd! --calibration !calibration!"
echo >>> !cmd!
!cmd!
goto menu

:backtest
set "ticker=AAPL"
set "period=1y"
set "interval=1h"
set "initial=10000"
set "seed=42"
set "longth=0.55"
set "shortth=0.40"
set "model=logistic"
set "calibration=platt"
set "adaptive=n"
set "export=y"
set "csv="

set /p input=Ticker [!ticker!]^> 
if not "!input!"=="" set "ticker=!input!"
set /p input=Periode (ex: 1y, 6mo) [!period!]^> 
if not "!input!"=="" set "period=!input!"
set /p input=Intervalle [!interval!]^> 
if not "!input!"=="" set "interval=!input!"
set /p input=Capital initial [!initial!]^> 
if not "!input!"=="" set "initial=!input!"
set /p input=Seed [!seed!]^> 
if not "!input!"=="" set "seed=!input!"
set /p input=Long threshold [!longth!]^> 
if not "!input!"=="" set "longth=!input!"
set /p input=Short threshold [!shortth!]^> 
if not "!input!"=="" set "shortth=!input!"
set /p input=Model (logistic/random_forest/xgboost) [!model!]^> 
if not "!input!"=="" set "model=!input!"
set /p input=Calibration (none/platt/isotonic) [!calibration!]^> 
if not "!input!"=="" set "calibration=!input!"
set /p input=Mode adaptatif ? (y/n) [!adaptive!]^> 
if not "!input!"=="" set "adaptive=!input!"
set /p input=Exporter CSV trades ? (y/n) [!export!]^> 
if not "!input!"=="" set "export=!input!"
if /I "!export!"=="y" (
    set "csv=data/trades_!ticker!_!period!.csv"
    set /p input=Chemin CSV [!csv!]^> 
    if not "!input!"=="" set "csv=!input!"
)

set "cmd=python scripts/run_backtest.py --ticker !ticker! --period !period! --initial-balance !initial! --seed !seed! --long-threshold !longth! --short-threshold !shortth!"
if not "!interval!"=="" set "cmd=!cmd! --interval !interval!"
if /I not "!model!"=="logistic" set "cmd=!cmd! --model-type !model!"
if /I not "!calibration!"=="none" set "cmd=!cmd! --calibration !calibration!"
if /I "!adaptive!"=="y" set "cmd=!cmd! --adaptive"
if not "!csv!"=="" set "cmd=!cmd! --export-trades !csv!"
echo >>> !cmd!
!cmd!
goto menu

:tests
pytest -q
goto menu

:custom
set /p custom=Commande (ex: python scripts/run_threshold_sweep.py --ticker AAPL)^> 
if "!custom!"=="" goto menu
echo >>> !custom!
!custom!
goto menu

:end
endlocal
exit /b 0
