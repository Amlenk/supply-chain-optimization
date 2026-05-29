# ==============================================================================
# DELOITTE DATA and AI PRACTICE -- SUPPLY CHAIN PIPELINE AUTOMATED SETUP
# Candidate: Aman Lenka
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "     DELOITTE SOUTH ASIA -- SUPPLY CHAIN ENGINE SETUP     " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Initializing production-grade ML forecasting environment...`n" -ForegroundColor Yellow

$projectDir = "C:\Users\amanl\.gemini\antigravity\scratch\supply-chain-optimization"
Set-Location $projectDir

# 1. CREATE VIRTUAL ENVIRONMENT
if (-not (Test-Path ".venv")) {
    Write-Host "[+] Creating local isolated Python virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
} else {
    Write-Host "[*] Python virtual environment (.venv) already exists." -ForegroundColor Gray
}

# 2. ACTIVATE VIRTUAL ENVIRONMENT
Write-Host "[+] Activating virtual environment..." -ForegroundColor Cyan
$envPath = Join-Path $projectDir ".venv\Scripts\Activate.ps1"
. $envPath

# Upgrade Pip
Write-Host "[+] Upgrading package manager (pip)..." -ForegroundColor Gray
python -m pip install --upgrade pip --quiet

# 3. INSTALL REQUIREMENTS
Write-Host "[+] Ingesting pipeline dependencies from requirements.txt..." -ForegroundColor Cyan
try {
    pip install -r requirements.txt
    Write-Host "[+] All core packages installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "[!] Unified requirements install encountered environment warnings. Executing individual fail-safe installs..." -ForegroundColor Yellow
    # Install robust fallbacks one by one (this ensures execution even if a specific package conflicts)
    pip install pandas numpy scikit-learn matplotlib streamlit requests --quiet
    try {
        pip install xgboost --quiet
        Write-Host "[+] XGBoost successfully compiled!" -ForegroundColor Green
    } catch {
        Write-Host "[!] XGBoost compilation warning. System will auto-fallback to robust RandomForest forecasting." -ForegroundColor DarkYellow
    }
}

# 4. EXECUTE PIPELINE STAGES
Write-Host "`n----------------------------------------------------------" -ForegroundColor Yellow
Write-Host "STAGE 1: Ingesting and Cleaning UCI Online Retail Dataset (500k+ Transactions)" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------" -ForegroundColor Yellow
python data_pipeline.py

Write-Host "`n----------------------------------------------------------" -ForegroundColor Yellow
Write-Host "STAGE 2: Training Predictive Time-Series Forecasting Models" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------" -ForegroundColor Yellow
python forecaster.py

Write-Host "`n----------------------------------------------------------" -ForegroundColor Yellow
Write-Host "STAGE 3: Simulating Operational Inventory Policy Optimization" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------" -ForegroundColor Yellow
python optimizer.py

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "          SETUP AND DATA PIPELINES COMPLETE!             " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "To launch your highly interactive Streamlit Web Dashboard:" -ForegroundColor Yellow
Write-Host "1. Open a new PowerShell terminal in this folder." -ForegroundColor White
Write-Host "2. Execute: .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "3. Execute: streamlit run app.py" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
