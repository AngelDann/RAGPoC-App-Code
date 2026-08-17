@echo off
title RAGPoC Studio
echo ===================================================
echo   Iniciando RAGPoC Studio para Windows...
echo ===================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3 no se encuentra en el PATH.
    echo Por favor instala Python 3.10 o superior desde python.org
    pause
    exit /b 1
)

:: Create virtualenv if not exists
if not exist ".venv" (
    echo Creando entorno virtual local...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Instalando dependencias necesarias...
    pip install -r requirements-windows.txt
) else (
    call .venv\Scripts\activate.bat
)

:: Copy env if missing
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo Se ha creado un archivo .env inicial. Puedes editarlo con tu clave de OpenRouter.
    )
)

:: Start RAGPoC Launcher
echo Iniciando servidor y abriendo navegador...
python desktop_launcher.py

pause
