#!/usr/bin/env bash
# POSIX counterpart of build.ps1: builds the C and Java programs of Assignment 1 into assignment1/build.
# Tools: $CC (default gcc), $JAVA_HOME (default: javac / java on PATH). Python needs no build step.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/build"
cc="${CC:-gcc}"
if [[ -n "${JAVA_HOME:-}" ]]; then javac="$JAVA_HOME/bin/javac"; java="$JAVA_HOME/bin/java"; else javac=javac; java=java; fi
exe=""; case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) exe=".exe";; esac
mkdir -p "$build/c" "$build/java"

cflags="-std=c11 -Wall -Wextra -O2"   # -O2, not -O3: GCC enables loop interchange at -O3
"$cc" $cflags -o "$build/c/matmul$exe" "$root/src/c/matmul.c"
"$javac" -d "$build/java" "$root/src/java/MatMul.java"

cat > "$build/build-info.json" <<EOF
{
  "c_compiler": "$(command -v "$cc")",
  "c_compiler_version": "$("$cc" --version | head -n1)",
  "c_flags": "$cflags",
  "javac": "$(command -v "$javac")",
  "java": "$(command -v "$java")",
  "java_version": "$("$java" -version 2>&1 | head -n1)"
}
EOF
echo "Built C and Java into $build"
