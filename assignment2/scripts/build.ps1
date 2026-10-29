# Builds the Assignment 2 C programs into assignment2/build (same compiler and -O2 flags as Assignment 1).
# C compiler: $env:CC, then gcc on PATH, then the WinLibs package installed with winget.
$ErrorActionPreference = 'Stop'
$root  = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build'
New-Item -ItemType Directory -Force $build | Out-Null

$cc = $env:CC
if (-not $cc) { $cmd = Get-Command gcc -ErrorAction SilentlyContinue; if ($cmd) { $cc = $cmd.Source } }
if (-not $cc) {
    $cc = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter gcc.exe -ErrorAction SilentlyContinue |
          Select-Object -First 1 -ExpandProperty FullName
}
if (-not $cc) { throw 'No C compiler found. Set $env:CC or install GCC (winget install BrechtSanders.WinLibs.POSIX.UCRT).' }

$cflags = @('-std=c11', '-Wall', '-Wextra', '-O2')
foreach ($prog in 'dense', 'sparse') {
    # -fopt-info-vec-optimized lists the loops GCC vectorised; kept as evidence (build/<prog>-vectorisation.txt)
    $info = Join-Path $build "$prog-vectorisation.txt"
    & $cc @cflags "-fopt-info-vec-optimized=$info" -o (Join-Path $build "$prog.exe") (Join-Path $root "src\c\$prog.c")
    if ($LASTEXITCODE -ne 0) { throw "C build failed for $prog" }
}

$info = [ordered]@{
    c_compiler         = $cc
    c_compiler_version = (& $cc --version | Select-Object -First 1)
    c_flags            = $cflags -join ' '
}
$json = $info | ConvertTo-Json
[System.IO.File]::WriteAllText((Join-Path $build 'build-info.json'), $json, (New-Object System.Text.UTF8Encoding($false)))
Write-Output "Built dense.exe and sparse.exe with $cc"
