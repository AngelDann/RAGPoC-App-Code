@echo off
title Compilador de RAGPoC para Windows (.EXE)
echo ===================================================
echo   Compilando RAGPoC.exe con PyInstaller
echo ===================================================

if not exist ".venv" (
    echo Creando entorno virtual local...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Instalando dependencias...
pip install -r requirements-windows.txt

echo Instalando PyInstaller...
pip install pyinstaller

echo Compilando ejecutable standalone...
pyinstaller ragpoc.spec --clean --noconfirm

echo ===================================================
echo   ¡Compilacion finalizada con exito!
echo   Tu ejecutable se encuentra en: dist\RAGPoC.exe
echo ===================================================
pause
