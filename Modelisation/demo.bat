@'
@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\demo.ps1"
pause
'@ | Set-Content -Path .\demo.bat -Encoding ASCII