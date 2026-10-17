# Builds the C and Java programs of Assignment 1 into assignment1/build.
#
# Tool discovery order:
#   C compiler : $env:CC, then gcc on PATH, then the WinLibs package from winget
#   JDK        : $env:JAVA_HOME, then a JDK under ~/.jdks, then javac on PATH
# Python needs no build step.
$ErrorActionPreference = 'Stop'
$root  = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build'
New-Item -ItemType Directory -Force (Join-Path $build 'c'), (Join-Path $build 'java') | Out-Null

# ---- C compiler ----
$cc = $env:CC
if (-not $cc) { $cmd = Get-Command gcc -ErrorAction SilentlyContinue; if ($cmd) { $cc = $cmd.Source } }
if (-not $cc) {
    $cc = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter gcc.exe -ErrorAction SilentlyContinue |
          Select-Object -First 1 -ExpandProperty FullName
}
if (-not $cc) { throw 'No C compiler found. Set $env:CC or install GCC (winget install BrechtSanders.WinLibs.POSIX.UCRT).' }

# ---- JDK ----
$javac = $null
if ($env:JAVA_HOME) { $javac = Join-Path $env:JAVA_HOME 'bin\javac.exe' }
if (-not $javac -or -not (Test-Path $javac)) {
    $javac = Get-ChildItem "$HOME\.jdks" -Recurse -Filter javac.exe -ErrorAction SilentlyContinue |
             Select-Object -First 1 -ExpandProperty FullName
}
if (-not $javac) { $cmd = Get-Command javac -ErrorAction SilentlyContinue; if ($cmd) { $javac = $cmd.Source } }
if (-not $javac) { throw 'No JDK found. Set $env:JAVA_HOME to a JDK 17 or newer.' }
$java = Join-Path (Split-Path $javac) 'java.exe'

# ---- C: -O2 (not -O3, which GCC allows to interchange loops) ----
$cflags = @('-std=c11', '-Wall', '-Wextra', '-O2')
& $cc @cflags -o (Join-Path $build 'c\matmul.exe') (Join-Path $root 'src\c\matmul.c')
if ($LASTEXITCODE -ne 0) { throw 'C build failed' }

# ---- Java ----
& $javac -d (Join-Path $build 'java') (Join-Path $root 'src\java\MatMul.java')
if ($LASTEXITCODE -ne 0) { throw 'Java build failed' }

# ---- record exactly what was used ----
# java -version prints to stderr; read it with errors non-fatal.
$javaVersion = & { $ErrorActionPreference = 'Continue'; & $java -version 2>&1 } | Select-Object -First 1
$info = [ordered]@{
    c_compiler         = $cc
    c_compiler_version = (& $cc --version | Select-Object -First 1)
    c_flags            = $cflags -join ' '
    javac              = $javac
    java               = $java
    java_version       = $javaVersion.ToString()
}
$json = $info | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText((Join-Path $build 'build-info.json'), $json, (New-Object System.Text.UTF8Encoding($false)))
Write-Output "Built C ($cc) and Java ($javac) into $build"
