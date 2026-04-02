#!/usr/bin/env python3
"""
run_tests.py — QuantLab QA Orchestrator
========================================
Single-command test runner that executes the full test suite and
writes a human-readable QA report to QA-REPORT.md.

Usage:
    python run_tests.py                  # run all tests, write QA-REPORT.md
    python run_tests.py --fast           # skip slow/integration tests
    python run_tests.py --module options # run only test_options.py
    python run_tests.py --out my_report.md
    python run_tests.py --no-cov         # skip coverage (faster)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).parent.resolve()
TESTS_DIR   = REPO_ROOT / "tests"
REPORT_PATH = REPO_ROOT / "QA-REPORT.md"
JUNIT_XML   = REPO_ROOT / "tests" / ".junit.xml"
COV_JSON    = REPO_ROOT / "tests" / ".coverage.json"

TEST_MODULES = {
    "valuation":   "tests/test_valuation.py",
    "portfolio":   "tests/test_portfolio.py",
    "options":     "tests/test_options.py",
    "bubble_ml":   "tests/test_bubble_ml.py",
    "risk_errors": "tests/test_risk_and_errors.py",
    "integration": "tests/test_integration.py",
}

MODULE_DESCRIPTIONS = {
    "valuation":   "Valuation Models (CAPM, Beta, WACC, DCF, Fama-French, APT)",
    "portfolio":   "Portfolio Optimization (9 strategies, risk matrices, bubble-aware)",
    "options":     "Options Pricing (Black-Scholes, Greeks, Payoff Diagrams)",
    "bubble_ml":   "Bubble Detection, Technical Indicators & ML Pipeline",
    "risk_errors": "Risk Score, Error Handling & Ticker Parser",
    "integration": "End-to-End Integration Pipeline",
}

GRADE_THRESHOLDS = {
    "A+": 100,
    "A":  97,
    "B":  90,
    "C":  80,
    "D":  70,
    "F":  0,
}

COVERAGE_THRESHOLDS = {
    "Excellent":  80,
    "Good":       60,
    "Acceptable": 40,
    "Low":        20,
    "Poor":       0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd=REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True
    )


def _grade(pass_pct: float) -> str:
    for grade, threshold in GRADE_THRESHOLDS.items():
        if pass_pct >= threshold:
            return grade
    return "F"


def _cov_label(pct: float) -> str:
    for label, threshold in COVERAGE_THRESHOLDS.items():
        if pct >= threshold:
            return label
    return "Poor"


def _status_emoji(passed: bool) -> str:
    return "✅" if passed else "❌"


def _bar(pct: float, width: int = 30) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _check_dependencies() -> list[str]:
    missing = []
    for pkg in ["pytest", "pytest_cov"]:
        r = _run([sys.executable, "-c", f"import {pkg}"])
        if r.returncode != 0:
            missing.append(pkg.replace("_", "-"))
    return missing


# ---------------------------------------------------------------------------
# Parsing pytest JSON / stdout output
# ---------------------------------------------------------------------------

def _parse_pytest_output(stdout: str) -> dict:
    """
    Parse pytest -v output into structured per-module and per-test results.
    Returns dict with keys: tests, passed, failed, error, skipped, duration.
    """
    results = {
        "tests": [],
        "passed": 0,
        "failed": 0,
        "error": 0,
        "skipped": 0,
        "duration": 0.0,
        "failures": [],
    }

    # Individual test lines: "tests/test_X.py::Class::method PASSED [  5%]"
    test_re = re.compile(
        r"(tests/\S+\.py)::(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)\s"
    )
    for line in stdout.splitlines():
        m = test_re.search(line)
        if m:
            module_path, test_name, status = m.groups()
            results["tests"].append({
                "module": module_path,
                "name": test_name,
                "status": status,
            })
            results[status.lower()] += 1

    # Summary line: "X passed, Y failed in Zs"
    summary_re = re.compile(r"(\d+) passed(?:, (\d+) failed)?.*?in ([\d.]+)s")
    sm = summary_re.search(stdout)
    if sm:
        results["passed"]  = int(sm.group(1))
        results["failed"]  = int(sm.group(2) or 0)
        results["duration"] = float(sm.group(3))

    # Collect failure details
    failure_section = False
    current_failure = []
    for line in stdout.splitlines():
        if line.startswith("FAILED ") or "FAILED tests/" in line:
            # Short summary line
            results["failures"].append(line.strip())
        if "_ _ _" in line or "======= FAILURES" in line:
            failure_section = True
            current_failure = []
        elif "======= short test summary" in line:
            failure_section = False
        elif failure_section:
            current_failure.append(line)

    return results


def _parse_coverage(json_path: Path) -> dict:
    """Parse coverage JSON export. Returns {file: pct, ...} and total."""
    if not json_path.exists():
        return {"total": 0.0, "files": {}}
    try:
        data = json.loads(json_path.read_text())
        total = data.get("totals", {}).get("percent_covered", 0.0)
        files = {
            k: v.get("summary", {}).get("percent_covered", 0.0)
            for k, v in data.get("files", {}).items()
            if not k.startswith("tests/")
        }
        return {"total": total, "files": files}
    except Exception:
        return {"total": 0.0, "files": {}}


def _per_module_stats(results: dict) -> dict:
    """Aggregate pass/fail counts per test module file."""
    stats = {}
    for t in results["tests"]:
        mod = t["module"]
        if mod not in stats:
            stats[mod] = {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "total": 0}
        stats[mod][t["status"].lower()] += 1
        stats[mod]["total"] += 1
    return stats


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(
    results: dict,
    coverage: dict,
    module_runs: dict,  # module_name -> (returncode, stdout, stderr)
    args: argparse.Namespace,
    report_path: Path,
) -> None:

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = results["passed"] + results["failed"] + results["error"]
    pass_pct = (results["passed"] / total * 100) if total > 0 else 0
    overall_passed = results["failed"] == 0 and results["error"] == 0
    grade = _grade(pass_pct)
    cov_total = coverage.get("total", 0.0)
    cov_label = _cov_label(cov_total)
    per_mod = _per_module_stats(results)

    lines = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "# QuantLab — QA Report",
        "",
        f"> Generated: **{now}**  |  "
        f"Grade: **{grade}**  |  "
        f"Coverage: **{cov_total:.1f}% ({cov_label})**",
        "",
    ]

    # ── Overall Status Banner ────────────────────────────────────────────────
    banner = "🟢 ALL TESTS PASSED" if overall_passed else "🔴 SOME TESTS FAILED"
    lines += [
        f"## {banner}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Tests | **{total}** |",
        f"| Passed | ✅ {results['passed']} |",
        f"| Failed | {'❌' if results['failed'] else '✅'} {results['failed']} |",
        f"| Errors | {'❌' if results['error'] else '✅'} {results['error']} |",
        f"| Skipped | ⏭️ {results['skipped']} |",
        f"| Pass Rate | {pass_pct:.1f}% `{_bar(pass_pct)}` |",
        f"| Duration | ⏱️ {results['duration']:.2f}s |",
        f"| Grade | **{grade}** |",
        "",
    ]

    # ── Coverage Summary ─────────────────────────────────────────────────────
    lines += [
        "## 📊 Coverage Summary",
        "",
        f"**Overall: {cov_total:.1f}%** `{_bar(cov_total)}` — {cov_label}",
        "",
    ]
    if coverage.get("files"):
        lines += [
            "| File | Coverage |",
            "|------|----------|",
        ]
        for fpath, pct in sorted(coverage["files"].items(),
                                  key=lambda x: x[1], reverse=True):
            icon = "🟢" if pct >= 60 else ("🟡" if pct >= 30 else "🔴")
            lines.append(f"| `{fpath}` | {icon} {pct:.1f}% `{_bar(pct, 20)}` |")
        lines.append("")

    # ── Module Results ───────────────────────────────────────────────────────
    lines += [
        "## 🧪 Test Modules",
        "",
        "| Module | Description | Tests | Passed | Failed | Status |",
        "|--------|-------------|-------|--------|--------|--------|",
    ]

    for key, path in TEST_MODULES.items():
        desc = MODULE_DESCRIPTIONS.get(key, "")
        mod_stats = per_mod.get(path, {"passed": 0, "failed": 0, "error": 0, "total": 0})
        p, f, t = mod_stats["passed"], mod_stats["failed"], mod_stats["total"]
        rc = module_runs.get(key, (0, "", ""))[0]
        status = _status_emoji(f == 0 and rc == 0)
        lines.append(f"| `{path}` | {desc} | {t} | {p} | {f} | {status} |")

    lines.append("")

    # ── Detailed Results Per Module ──────────────────────────────────────────
    lines += ["## 📋 Detailed Test Results", ""]

    for key, path in TEST_MODULES.items():
        desc = MODULE_DESCRIPTIONS[key]
        rc, stdout, _ = module_runs.get(key, (0, "", ""))
        mod_tests = [t for t in results["tests"] if t["module"] == path]
        n_passed = sum(1 for t in mod_tests if t["status"] == "PASSED")
        n_failed = sum(1 for t in mod_tests if t["status"] in ("FAILED", "ERROR"))
        status_hdr = "✅ PASSED" if n_failed == 0 else f"❌ {n_failed} FAILED"

        lines += [
            f"### {desc}",
            f"**File:** `{path}` &nbsp;|&nbsp; **Status:** {status_hdr} "
            f"&nbsp;|&nbsp; **{n_passed}/{len(mod_tests)} tests passing**",
            "",
        ]

        if mod_tests:
            lines += [
                "<details>",
                "<summary>Show all tests</summary>",
                "",
                "| Test | Status |",
                "|------|--------|",
            ]
            for t in mod_tests:
                icon = _status_emoji(t["status"] == "PASSED")
                lines.append(f"| `{t['name']}` | {icon} {t['status']} |")
            lines += ["", "</details>", ""]

    # ── Failures Detail ──────────────────────────────────────────────────────
    if results["failures"]:
        lines += [
            "## ❌ Failure Details",
            "",
            "> Expand each module's section above to see per-test status.",
            "> Run `pytest tests/ -v --tb=long` locally for full tracebacks.",
            "",
            "| # | Failed Test |",
            "|---|------------|",
        ]
        for i, f in enumerate(results["failures"], 1):
            lines.append(f"| {i} | `{f}` |")
        lines.append("")

    # ── How to Run ───────────────────────────────────────────────────────────
    lines += [
        "## 🚀 Running the Tests",
        "",
        "### Quick start",
        "```bash",
        "# Install dependencies",
        "pip install -r requirements.txt",
        "pip install pytest pytest-cov",
        "",
        "# Run everything and regenerate this report",
        "python run_tests.py",
        "```",
        "",
        "### Options",
        "```bash",
        "python run_tests.py --fast           # skip integration tests",
        "python run_tests.py --module options  # run only test_options.py",
        "python run_tests.py --no-cov         # skip coverage (faster)",
        "python run_tests.py --out my_qa.md   # custom output path",
        "```",
        "",
        "### Direct pytest",
        "```bash",
        "pytest tests/ -v                      # verbose output",
        "pytest tests/ -v --tb=long            # full failure tracebacks",
        "pytest tests/ -k 'test_capm'          # run matching tests only",
        "pytest tests/ --cov=. --cov-report=html  # HTML coverage report",
        "```",
        "",
        "### Makefile shortcuts",
        "```bash",
        "make test     # run all tests",
        "make qa       # full suite + report",
        "make fast     # skip integration",
        "make coverage # open HTML coverage report",
        "make clean    # remove test artifacts",
        "```",
        "",
    ]

    # ── Test Architecture ────────────────────────────────────────────────────
    lines += [
        "## 🏗️ Test Architecture",
        "",
        "```",
        "tests/",
        "├── conftest.py            # Streamlit stub + shared fixtures",
        "│                          # (synthetic 2-year OHLCV data — no network needed)",
        "├── test_valuation.py      # CAPM, Beta (ddof=1 verified), WACC, DCF, FF, APT",
        "├── test_portfolio.py      # 9 strategies × 3 invariants (parametrized)",
        "│                          # Risk Parity contributions, HRP, bubble penalty",
        "├── test_options.py        # B-S known value, put-call parity, all 5 Greeks",
        "│                          # Payoff floors/caps for 6 strategies",
        "├── test_bubble_ml.py      # BubbleDetector, GPH SE, RSI Wilder's EMA,",
        "│                          # MACD histogram, ML pipeline, sentiment",
        "├── test_risk_and_errors.py # Risk score VIX/yield/gold, exception hierarchy,",
        "│                          # handle_error decorator, ticker parser, fetch errors",
        "└── test_integration.py    # End-to-end: prices → metrics → portfolio →",
        "                           # bubble → technicals → ML → options → narrative",
        "```",
        "",
        "All tests run **fully offline** using deterministic synthetic data.",
        "No Yahoo Finance calls are made during testing.",
        "",
    ]

    # ── Footer ───────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        f"*Report generated by `run_tests.py` on {now}*",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="QuantLab QA Orchestrator — run tests and generate a report."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip integration tests (faster CI check)."
    )
    parser.add_argument(
        "--module", metavar="NAME", default=None,
        choices=list(TEST_MODULES.keys()),
        help=f"Run only one module. Choices: {', '.join(TEST_MODULES)}"
    )
    parser.add_argument(
        "--out", metavar="PATH", default=str(REPORT_PATH),
        help="Output report path (default: QA-REPORT.md)"
    )
    parser.add_argument(
        "--no-cov", action="store_true",
        help="Disable coverage collection (faster)."
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Run tests but do not write a report file."
    )
    args = parser.parse_args()
    report_path = Path(args.out)

    # ── Banner ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  QuantLab QA Orchestrator")
    print("═" * 60)
    print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Report  : {report_path}")
    print(f"  Mode    : {'fast (no integration)' if args.fast else 'full suite'}")
    print("═" * 60 + "\n")

    # ── Dependency check ─────────────────────────────────────────────────────
    missing = _check_dependencies()
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}\n")
        install = input("Install now? [y/N] ").strip().lower()
        if install == "y":
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing,
                           check=True)
        else:
            sys.exit(1)

    # ── Determine which modules to run ───────────────────────────────────────
    if args.module:
        modules_to_run = {args.module: TEST_MODULES[args.module]}
    elif args.fast:
        modules_to_run = {k: v for k, v in TEST_MODULES.items() if k != "integration"}
    else:
        modules_to_run = TEST_MODULES

    # ── Run per-module (for per-module status) ────────────────────────────────
    module_runs = {}
    print("Running test modules:")
    for key, path in modules_to_run.items():
        desc = MODULE_DESCRIPTIONS[key]
        print(f"  ▶  {desc}", end=" ... ", flush=True)
        cmd = [
            sys.executable, "-m", "pytest", path,
            "-v", "--tb=short", "--no-header", "-q",
            "--no-cov",        # per-module runs skip cov (collected at full run)
        ]
        r = _run(cmd)
        icon = "✅" if r.returncode == 0 else "❌"
        # Parse pass count from output
        sm = re.search(r"(\d+) passed", r.stdout)
        fm = re.search(r"(\d+) failed", r.stdout)
        p_cnt = int(sm.group(1)) if sm else 0
        f_cnt = int(fm.group(1)) if fm else 0
        print(f"{icon}  {p_cnt} passed, {f_cnt} failed")
        module_runs[key] = (r.returncode, r.stdout, r.stderr)

    # ── Full combined run (for accurate totals + coverage) ───────────────────
    print("\nRunning full suite for totals and coverage...")
    paths = list(modules_to_run.values())
    cmd = [
        sys.executable, "-m", "pytest",
        *paths,
        "-v", "--tb=short", "--no-header",
    ]
    if not args.no_cov:
        cmd += [
            f"--cov={REPO_ROOT}",
            "--cov-report=json:" + str(COV_JSON),
            "--cov-report=html:tests/coverage_html",
            "--cov-config=.coveragerc",
        ]
    else:
        cmd.append("--no-cov")

    full_run = _run(cmd)
    results = _parse_pytest_output(full_run.stdout)
    coverage = _parse_coverage(COV_JSON) if not args.no_cov else {"total": 0.0, "files": {}}

    # ── Console summary ──────────────────────────────────────────────────────
    total = results["passed"] + results["failed"] + results["error"]
    pass_pct = (results["passed"] / total * 100) if total > 0 else 0
    grade = _grade(pass_pct)
    print()
    print("═" * 60)
    print(f"  Results  : {results['passed']}/{total} passed  ({pass_pct:.1f}%)")
    print(f"  Grade    : {grade}")
    if not args.no_cov:
        print(f"  Coverage : {coverage['total']:.1f}% ({_cov_label(coverage['total'])})")
    print(f"  Duration : {results['duration']:.2f}s")
    print("═" * 60)

    if results["failures"]:
        print(f"\n❌  {len(results['failures'])} test(s) failed:")
        for f in results["failures"]:
            print(f"    • {f}")

    # ── Write report ─────────────────────────────────────────────────────────
    if not args.no_report:
        _generate_report(results, coverage, module_runs, args, report_path)
        print(f"\n📄 Report written → {report_path}")
        if not args.no_cov:
            print(f"📊 HTML coverage  → tests/coverage_html/index.html")

    # ── Exit code ────────────────────────────────────────────────────────────
    overall_ok = results["failed"] == 0 and results["error"] == 0
    if not overall_ok:
        print("\n💡 Run `pytest tests/ -v --tb=long` for full failure details.")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
