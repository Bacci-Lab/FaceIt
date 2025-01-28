# Ensure PowerShell script stops on errors
$ErrorActionPreference = "Stop"

# Check if pyenv-win is installed
if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) {
    Write-Host "🚀 pyenv not found. Installing pyenv-win..."
    Invoke-Expression "& {iwr -useb https://pyenv.run} | Invoke-Expression"

    # Restart the terminal to apply changes
    Write-Host "🔄 Restarting shell to apply pyenv..."
    exit
}

# Set the desired Python version
$pythonVersion = "3.9.13"

# Install and set the correct Python version
Write-Host "🐍 Ensuring Python $pythonVersion is installed..."
pyenv install $pythonVersion --skip-existing
pyenv local $pythonVersion

# Verify Python version
Write-Host "✅ Using Python version:"
pyenv version

# Wait for Python to be properly set
Start-Sleep -Seconds 2

# Create a virtual environment (if not already created)
if (-not (Test-Path ".venv")) {
    Write-Host "🔧 Creating virtual environment..."
    python -m venv .venv
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

# Install Poetry if not already installed
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Host "📦 Poetry not found. Installing Poetry..."
    Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing | python -

    # Ensure Poetry is available in the current session
    $env:PATH += ";$HOME\.local\bin"
}

# Reload the terminal to apply Poetry installation
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Host "⚠️ Poetry installation not detected. Restart your terminal and run this script again."
    exit
}

# Install dependencies using Poetry
Write-Host "📥 Installing dependencies..."
poetry install

# Ensure installation was successful
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: Dependency installation failed. Please check the error message above."
    exit 1
}

Write-Host "✅ Setup complete! Your environment is ready."

# Auto-activate virtual environment for the user
Write-Host "🔄 Activating environment for immediate use..."
& .\.venv\Scripts\Activate.ps1

Write-Host "🚀 Run the program with:"
Write-Host "poetry run python main.py"
