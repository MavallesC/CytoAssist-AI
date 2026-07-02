@echo off
echo ========================================================
echo        CytoAssist AI - Script de Configuracion Rapida
echo ========================================================
echo.

echo 1. Instalando dependencias de Python desde requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Hubo un problema al instalar las dependencias.
    pause
    exit /b %errorlevel%
)
echo.

echo 2. Ejecutando migraciones de la base de datos...
python manage.py makemigrations wsi_analysis
python manage.py migrate
if %errorlevel% neq 0 (
    echo [ERROR] Hubo un problema al ejecutar las migraciones.
    pause
    exit /b %errorlevel%
)
echo.

echo 3. Creando directorios necesarios...
if not exist "media" mkdir "media"
if not exist "media\wsi" mkdir "media\wsi"
if not exist "media\results" mkdir "media\results"
if not exist "static" mkdir "static"
if not exist "static\css" mkdir "static\css"
if not exist "artefactos" mkdir "artefactos"
echo.

echo ========================================================
echo    Configuracion finalizada.
echo    Asegurese de colocar sus artefactos de IA en:
echo    %CD%\artefactos\
echo ========================================================
echo.
echo Iniciando servidor web de Django...
python manage.py runserver
pause
