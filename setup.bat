@echo off
setlocal
cd /d "%~dp0"
set "PYTHON="
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PYTHON=%LocalAppData%\Programs\Python\Python311\python.exe"
if not defined PYTHON for /f "delims=" %%I in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON=%%I"
if not defined PYTHON (echo Python 3.11 or 3.12 x64 is required.& pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" "%PYTHON%" -m venv .venv
if errorlevel 1 (echo Could not create virtual environment.& pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto :fail
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail
python -c "from transformers import AutoModelForImageSegmentation as A; A.from_pretrained('ZhengPeng7/BiRefNet-portrait', revision='ecdeb6240ef23557dbd48ff27c59c1a88cbcb755', trust_remote_code=True); print('Portrait model downloaded')"
if errorlevel 1 goto :fail
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam2.1-hiera-small', revision='ee5bba1d82bb8749febdf90f45e84b687142ba03'); print('SAM 2.1 verifier downloaded')"
if errorlevel 1 goto :fail
python -c "from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights as W; W.DEFAULT.get_state_dict(progress=True); print('Person QC model downloaded')"
if errorlevel 1 goto :fail
python smoke_test.py
if errorlevel 1 goto :fail
echo.
echo Setup completed successfully. Start with "Background Remover.bat".
pause
exit /b 0
:fail
echo.
echo Setup failed. Run diagnose.bat and save its output.
pause
exit /b 1
