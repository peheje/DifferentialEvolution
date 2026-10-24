#!/usr/bin/env bash
# Builds and benchmarks the canonical DE in C, C# and Roc (single-threaded,
# same algorithm, RNG and seed). Prints the median of N runs per variant.
set -euo pipefail
cd "$(dirname "$0")"
N=${N:-7}

echo "building..."
gcc -O3 -march=native -ffp-contract=off -o c/de_c c/de.c
dotnet build -c Release csharp -o csharp/out -v q -nologo >/dev/null
roc build --optimize roc/main.roc --output roc/de_roc >/dev/null
roc build --optimize roc/main_inplace.roc --output roc/de_roc_inplace >/dev/null

median() { sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }

run() {
    local name=$1 bin=$2 score
    local times=()
    for _ in $(seq "$N"); do
        out=$("$bin")
        score=$(sed -n 's/^score //p' <<<"$out")
        times+=("$(sed -n 's/^execution time \([0-9]*\) ms/\1/p' <<<"$out")")
    done
    printf "%-22s %6s ms   score %s\n" "$name" "$(printf '%s\n' "${times[@]}" | median)" "$score"
}

echo "median of $N runs:"
run "C (gcc -O3)"          ./c/de_c
run "C# (.NET 10)"         ./csharp/out/de_csharp
run "Roc (simple)"         ./roc/de_roc
run "Roc (in-place)"       ./roc/de_roc_inplace
