@echo off
echo ==========================================
echo Customers Segmentation - Setup
echo ==========================================

REM Create virtual environment
echo Creating virtual environment...
python -m venv customers-segmentation

REM Activate virtual environment
echo Activating virtual environment...
call customers-segmentation\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
if not exist "data" mkdir data
if not exist "artifacts" mkdir artifacts

echo ==========================================
echo Setup completed successfully!
echo ==========================================
pause