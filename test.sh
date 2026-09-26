#!/usr/bin/env bash
################################################################################
# spawnllm test suite — the body of `make test`, the one gate.
#
# Runs, in order, and always all of them (a failure is recorded, not fatal):
#   1. format   golangci-lint fmt --diff   (verifies only; never rewrites)
#   2. vet      go vet ./...
#   3. lint     golangci-lint run ./...
#   4. test     go test -race -count=1 -json ./...
# then prints a summary: test totals and per-stage PASS/FAIL.
#
# Usage:
#   ./test.sh              run the suite
#   ./test.sh -k|--keep    keep the raw go test JSON output (path is printed)
#   ./test.sh -n|--no-color
#   ./test.sh -h|--help
#
# The working tree is never modified: the only artifact is the go test JSON
# stream, written to a temporary directory that is removed on exit unless -k.
#
# golangci-lint is required. It is looked up as $GOLANGCI_LINT, then on PATH,
# then in $(go env GOPATH)/bin. If it is missing the suite fails and says how
# to install it; it is never installed for you.
#
# Exit codes: 0 when every stage passed, 1 otherwise.
################################################################################

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

GOLANGCI_LINT_INSTALL="go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest"

KEEP=false
COLOR_OFF=false

usage() {
    echo "Usage: $0 [-k|--keep] [-n|--no-color] [-h|--help]"
    echo "  -k, --keep      keep the go test JSON output for debugging"
    echo "  -n, --no-color  disable colour output"
    echo "  -h, --help      show this help"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -k|--keep) KEEP=true ;;
        -n|--no-color) COLOR_OFF=true ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

# Colours off for -n, NO_COLOR, or when stdout is not a terminal.
if $COLOR_OFF || [ -n "${NO_COLOR+x}" ] || [ ! -t 1 ]; then
    RED='' GREEN='' YELLOW='' BOLD='' DIM='' NC=''
else
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[1;33m'
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
    NC=$'\033[0m'
fi

if ! command -v go >/dev/null 2>&1; then
    echo "${RED}ERROR: go not found in PATH${NC}"
    exit 1
fi

if [ -z "${GOLANGCI_LINT:-}" ]; then
    GOLANGCI_LINT="$(command -v golangci-lint 2>/dev/null || echo "$(go env GOPATH)/bin/golangci-lint")"
fi
HAVE_LINT=true
if [ ! -x "$GOLANGCI_LINT" ]; then
    HAVE_LINT=false
fi

WORK_DIR="$(mktemp -d)"
TEST_JSON="$WORK_DIR/go-test.json"
if $KEEP; then
    trap 'echo "${DIM}Kept: ${TEST_JSON}${NC}"' EXIT
else
    trap 'rm -rf "$WORK_DIR"' EXIT
fi

declare -a STAGE_NAMES=()
declare -a STAGE_RESULTS=()
OVERALL=0

# record NAME STATUS: remember a stage's outcome for the summary.
record() {
    STAGE_NAMES+=("$1")
    if [ "$2" -eq 0 ]; then
        STAGE_RESULTS+=("PASS")
        echo "${GREEN}PASS${NC} $1"
    else
        STAGE_RESULTS+=("FAIL")
        echo "${RED}FAIL${NC} $1"
        OVERALL=1
    fi
    echo ""
}

lint_missing() {
    echo "${RED}ERROR: golangci-lint not found (looked for ${GOLANGCI_LINT}).${NC}"
    echo "Install it with: ${GOLANGCI_LINT_INSTALL}"
    echo "or set GOLANGCI_LINT=/path/to/golangci-lint"
}

echo ""
echo "${BOLD}spawnllm test suite${NC}  ${DIM}($(go version | awk '{print $3}'))${NC}"
echo ""

# 1. Formatting: verify only. --diff prints what `make fmt` would change and
# exits non-zero when anything would.
echo "${BOLD}[1/4] format (golangci-lint fmt --diff)${NC}"
if $HAVE_LINT; then
    fmt_out="$("$GOLANGCI_LINT" fmt --diff 2>&1)"
    status=$?
    if [ -n "$fmt_out" ]; then
        echo "$fmt_out"
        echo "${YELLOW}Run 'make fmt' to fix formatting.${NC}"
        status=1
    fi
else
    lint_missing
    status=1
fi
record "format" "$status"

# 2. go vet
echo "${BOLD}[2/4] vet (go vet ./...)${NC}"
go vet ./...
record "vet" $?

# 3. golangci-lint
echo "${BOLD}[3/4] lint (golangci-lint run ./...)${NC}"
if $HAVE_LINT; then
    "$GOLANGCI_LINT" run ./...
    status=$?
else
    lint_missing
    status=1
fi
record "lint" "$status"

# 4. go test with the race detector. -json gives exact per-test outcomes; the
# stream is summarised below rather than shown raw.
echo "${BOLD}[4/4] test (go test -race -count=1 ./...)${NC}"
echo "${DIM}(race detector on; the WaitDelay tests take a few seconds each)${NC}"
go test -race -count=1 -json ./... >"$TEST_JSON" 2>"$WORK_DIR/go-test.stderr"
test_status=$?

# Parse the test2json stream with POSIX awk (no jq dependency). Emits:
#   PKG <status> <package>          one per package result
#   FAILED <package> <test>         one per failed test
#   COUNTS <pass> <fail> <skip>     totals over all tests, subtests included
# and, for each failed test, its captured output prefixed with "OUT ".
parsed="$(awk '
    function field(name,   re, v) {
        re = "\"" name "\":\"([^\"\\\\]|\\\\.)*\""
        if (!match($0, re)) return ""
        v = substr($0, RSTART + length(name) + 4, RLENGTH - length(name) - 5)
        return v
    }
    function unescape(s,   out, i, c, n, hex) {
        out = ""
        n = length(s)
        for (i = 1; i <= n; i++) {
            c = substr(s, i, 1)
            if (c != "\\" || i == n) { out = out c; continue }
            c = substr(s, ++i, 1)
            if (c == "n") out = out "\n"
            else if (c == "t") out = out "\t"
            else if (c == "r") continue
            else if (c == "u") {
                # Go escapes <, > and & as <, >, &.
                hex = substr(s, i + 1, 4); i += 4
                if (hex == "003c") out = out "<"
                else if (hex == "003e") out = out ">"
                else if (hex == "0026") out = out "&"
                else out = out "\\u" hex
            }
            else out = out c
        }
        return out
    }
    {
        action = field("Action"); pkg = field("Package"); test = field("Test")
        if (action == "build-output") {
            buildout = buildout unescape(field("Output"))
            next
        }
        if (action == "output" && test != "") {
            key = pkg SUBSEP test
            out[key] = out[key] unescape(field("Output"))
            next
        }
        if (action == "output" && test == "") {
            pkgout[pkg] = pkgout[pkg] unescape(field("Output"))
            next
        }
        if (action != "pass" && action != "fail" && action != "skip") next
        if (test == "") {
            print "PKG " action " " pkg
            if (action == "fail") pkgfail[pkg] = 1
            next
        }
        if (action == "pass") pass++
        else if (action == "skip") skip++
        else { fail++; failed[++nf] = pkg SUBSEP test }
    }
    END {
        for (i = 1; i <= nf; i++) {
            split(failed[i], parts, SUBSEP)
            print "FAILED " parts[1] " " parts[2]
            n = split(out[failed[i]], lines, "\n")
            for (j = 1; j <= n; j++) if (lines[j] != "") print "OUT     " lines[j]
        }
        # Package-level output of a failed package (panics outside a test,
        # TestMain), minus the FAIL/ok summary lines go test adds itself.
        for (p in pkgfail) {
            n = split(pkgout[p], lines, "\n")
            for (j = 1; j <= n; j++) {
                if (lines[j] == "" || lines[j] ~ /^(FAIL|ok|PASS)([ \t]|$)/) continue
                print "OUT " p ": " lines[j]
            }
        }
        # Compiler errors for packages that failed to build.
        n = split(buildout, lines, "\n")
        for (j = 1; j <= n; j++) if (lines[j] != "") print "OUT " lines[j]
        printf "COUNTS %d %d %d\n", pass, fail, skip
    }
' "$TEST_JSON")"

read -r _ PASS_COUNT FAIL_COUNT SKIP_COUNT <<<"$(grep '^COUNTS ' <<<"$parsed")"
TOTAL_COUNT=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))

while read -r _ result pkg; do
    [ -z "${pkg:-}" ] && continue
    case "$result" in
        pass) echo "  ${GREEN}ok${NC}    $pkg" ;;
        skip) echo "  ${DIM}--    $pkg (no tests)${NC}" ;;
        fail) echo "  ${RED}FAIL${NC}  $pkg" ;;
    esac
done < <(grep '^PKG ' <<<"$parsed")

if [ "$test_status" -ne 0 ]; then
    echo ""
    echo "${RED}${BOLD}Failure details${NC}"
    grep -E '^(FAILED|OUT) ' <<<"$parsed" | sed -E "s/^FAILED (.*)/${RED}--- FAIL: \1${NC}/; s/^OUT //"
    if [ -s "$WORK_DIR/go-test.stderr" ]; then
        cat "$WORK_DIR/go-test.stderr"
    fi
elif [ "$FAIL_COUNT" -ne 0 ]; then
    # Defensive: a failed test with a zero exit should be impossible.
    test_status=1
fi
echo ""
record "test" "$test_status"

# Summary
echo "${BOLD}============================================${NC}"
echo "${BOLD}   SUMMARY${NC}"
echo "${BOLD}============================================${NC}"
echo "Tests: ${TOTAL_COUNT}  ${GREEN}passed: ${PASS_COUNT}${NC}  ${RED}failed: ${FAIL_COUNT}${NC}  ${YELLOW}skipped: ${SKIP_COUNT}${NC}"
echo ""
for i in "${!STAGE_NAMES[@]}"; do
    if [ "${STAGE_RESULTS[$i]}" = "PASS" ]; then
        printf "  %-8s %s\n" "${STAGE_NAMES[$i]}" "${GREEN}PASS${NC}"
    else
        printf "  %-8s %s\n" "${STAGE_NAMES[$i]}" "${RED}FAIL${NC}"
    fi
done
echo ""
if [ "$OVERALL" -eq 0 ]; then
    echo "${GREEN}${BOLD}All stages passed.${NC}"
else
    echo "${RED}${BOLD}FAILURES DETECTED${NC}"
fi
exit "$OVERALL"
