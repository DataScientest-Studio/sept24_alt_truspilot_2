Write-Host ""
Write-Host "=== Lancement de la stack Trustpilot MLOps ===" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

docker compose down --remove-orphans
docker compose up -d --build

Write-Host ""
Write-Host "=== Etat des conteneurs ===" -ForegroundColor Cyan
docker compose ps

Write-Host ""
Write-Host "=== URLs utiles ===" -ForegroundColor Green
Write-Host "Streamlit   : http://127.0.0.1:8501"
Write-Host "FastAPI     : http://127.0.0.1:8001/docs"
Write-Host "MLflow      : http://127.0.0.1:5001"
Write-Host "Airflow     : http://127.0.0.1:8080"
Write-Host "Prometheus  : http://127.0.0.1:9090"
Write-Host "Grafana     : http://127.0.0.1:3000"
Write-Host ""
Write-Host "Grafana local : admin / admin"
Write-Host ""
Write-Host "Stack lancee." -ForegroundColor Green
