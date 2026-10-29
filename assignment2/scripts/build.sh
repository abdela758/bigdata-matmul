#!/usr/bin/env bash
# POSIX counterpart of build.ps1 (not run on the test machine).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; build="$root/build"; cc="${CC:-gcc}"
exe=""; case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) exe=".exe";; esac
mkdir -p "$build"
cflags="-std=c11 -Wall -Wextra -O2"
for prog in dense sparse; do
  "$cc" $cflags "-fopt-info-vec-optimized=$build/$prog-vectorisation.txt" -o "$build/$prog$exe" "$root/src/c/$prog.c"
done
printf '{\n  "c_compiler": "%s",\n  "c_compiler_version": "%s",\n  "c_flags": "%s"\n}\n' "$(command -v "$cc")" "$("$cc" --version | head -n1)" "$cflags" > "$build/build-info.json"
echo "Built dense and sparse into $build"
