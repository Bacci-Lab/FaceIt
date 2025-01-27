# Check if pyenv-win is installed
if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) {
    Write-Host "pyenv not found. Installing pyenv-win..."
    Invoke-Expression "& {iwr -useb https://pyenv.run} | Invoke-Expression"
}

# Set the desired Python version
$pythonVersion = "3.9.13"

# Install and set the correct Python version
Write-Host "Ensuring Python $pythonVersion is installed..."
pyenv install $pythonVersion -SkipExisting
pyenv local $pythonVersion

# Verify Python version
Write-Host "Using Python version:"
pyenv version

# Create a virtual environment (if not already created)
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

# Activate the virtual environment
Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

# Install Poetry if not already installed
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Host "Poetry not found. Installing Poetry..."
    Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing | python -
    $env:PATH += ";$HOME\.local\bin"
}

# Install dependencies using Poetry
Write-Host "Installing dependencies..."
poetry install

Write-Host "Setup complete! Use the following command to activate your environment:"
Write-Host ".\\.venv\\Scripts\\Activate.ps1"
