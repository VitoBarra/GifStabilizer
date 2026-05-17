param(
    [string]$PythonExe = ".\venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $projectRoot

try {
    $pythonCommand = Get-Command $PythonExe -ErrorAction Stop
} catch {
    throw "Python executable not found: $PythonExe"
}

$pythonTarget = $pythonCommand.Source

$distRoot = Join-Path $projectRoot "dist"
$buildRoot = Join-Path $projectRoot "build"
$logoPng = Join-Path $projectRoot "assets\branding\GifStabilizer-logo.png"
$logoIco = Join-Path $projectRoot "assets\branding\GifStabilizer-logo.ico"

if (-not (Test-Path -LiteralPath $logoPng)) {
    throw "Logo PNG not found: $logoPng"
}
if (-not (Test-Path -LiteralPath $logoIco)) {
    throw "Logo ICO not found: $logoIco"
}

Write-Host "Cleaning previous build output..."
if (Test-Path -LiteralPath $distRoot) {
    Remove-Item -LiteralPath $distRoot -Recurse -Force
}
if (Test-Path -LiteralPath $buildRoot) {
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
}

$commonArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--paths", $projectRoot,
    "--distpath", $distRoot,
    "--workpath", $buildRoot
)

Write-Host "Building GUI executable..."
& $pythonTarget @commonArgs `
    "--onefile" `
    "--windowed" `
    "--icon" $logoIco `
    "--add-data" "$logoPng;assets\branding" `
    "--name" "GifStabilizer-GUI" `
    "App\aligngif_gui.py"

Write-Host "Building CLI executable..."
& $pythonTarget @commonArgs `
    "--onefile" `
    "--console" `
    "--icon" $logoIco `
    "--name" "GifStabilizer-CLI" `
    "App\aligngif_cli.py"

Write-Host ""
Write-Host "Build completed."
Write-Host "GUI artifact: $distRoot\GifStabilizer-GUI.exe"
Write-Host "CLI artifact: $distRoot\GifStabilizer-CLI.exe"
Write-Host "These executables are self-contained."
