#!/usr/bin/env bash
# Build script for hyprl_supercalc Rust extension

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NATIVE_DIR="$PROJECT_ROOT/native/hyprl_supercalc"

echo "==> Building hyprl_supercalc Rust extension..."
echo "    Native directory: $NATIVE_DIR"

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "ERROR: Rust/Cargo not found. Please install from https://rustup.rs/"
    exit 1
fi

cd "$NATIVE_DIR"

# Build in release mode
echo "==> Running: cargo build --release"
cargo build --release

# Find the built library
LIB_PATH=$(find target/release -name "libhyprl_supercalc.so" -o -name "hyprl_supercalc.dylib" -o -name "hyprl_supercalc.dll" | head -n 1)

if [ -z "$LIB_PATH" ]; then
    echo "ERROR: Could not find built library"
    exit 1
fi

echo "==> Built library: $LIB_PATH"

# Copy to Python package location
DEST_DIR="$PROJECT_ROOT/src/hyprl"
DEST_NAME="hyprl_supercalc$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

if [ -z "$DEST_NAME" ]; then
    # Fallback for simpler naming
    case "$(uname -s)" in
        Linux*)     DEST_NAME="hyprl_supercalc.so";;
        Darwin*)    DEST_NAME="hyprl_supercalc.so";;
        *)          DEST_NAME="hyprl_supercalc.pyd";;
    esac
fi

echo "==> Copying to: $DEST_DIR/$DEST_NAME"
cp "$LIB_PATH" "$DEST_DIR/$DEST_NAME"

echo "==> Build complete!"
echo "    You can now import hyprl_supercalc in Python"
