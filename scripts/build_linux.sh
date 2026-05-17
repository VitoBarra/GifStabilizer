#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${1:-./venv/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

if ! command -v "$PYTHON_EXE" >/dev/null 2>&1 && [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Python executable not found: $PYTHON_EXE" >&2
  exit 1
fi

if command -v "$PYTHON_EXE" >/dev/null 2>&1; then
  PYTHON_TARGET="$(command -v "$PYTHON_EXE")"
else
  PYTHON_TARGET="$PYTHON_EXE"
fi

DIST_ROOT="$PROJECT_ROOT/dist"
BUILD_ROOT="$PROJECT_ROOT/build"
LOGO_PNG="$PROJECT_ROOT/assets/branding/GifStabilizer-logo.png"

if [[ ! -f "$LOGO_PNG" ]]; then
  echo "Logo PNG not found: $LOGO_PNG" >&2
  exit 1
fi

echo "Cleaning previous build output..."
rm -rf "$DIST_ROOT" "$BUILD_ROOT"

COMMON_ARGS=(
  -m PyInstaller
  --noconfirm
  --clean
  --paths "$PROJECT_ROOT"
  --distpath "$DIST_ROOT"
  --workpath "$BUILD_ROOT"
)

echo "Building GUI executable..."
"$PYTHON_TARGET" "${COMMON_ARGS[@]}" \
  --onefile \
  --windowed \
  --add-data "$LOGO_PNG:assets/branding" \
  --name GifStabilizer-GUI \
  App/aligngif_gui.py

echo "Building CLI executable..."
"$PYTHON_TARGET" "${COMMON_ARGS[@]}" \
  --onefile \
  --console \
  --name GifStabilizer-CLI \
  App/aligngif_cli.py

echo
echo "Build completed."
echo "GUI artifact: $DIST_ROOT/GifStabilizer-GUI"
echo "CLI artifact: $DIST_ROOT/GifStabilizer-CLI"
echo "These executables are self-contained."
