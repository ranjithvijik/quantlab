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
import xml.etree.ElementTree as ET
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

GRADE_THRESHOLDS = [
    ("A+", 100), ("A", 97), ("B", 90), ("C", 80), ("D", 70), ("F", 0),
]

# Coverage thresholds recalibrated for Streamlit apps:
# Only the pure-Python logic (~20% of app.py) is exercised by offline tests.
COVERAGE_THRESHOLDS = [
    ("Excellent", 50), ("Good", 30), ("Acceptable", 18), ("Low", 10), ("Poor", 0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, cwd=REPO_ROOT):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*[mK]', '', text)


def _grade(pass_pct):
    for grade, threshold in GRADE_THRESHOLDS:
        if pass_pct >= threshold:
            return grade
    return "F"


def _cov_label(pct):
    for label, threshold in COVERAGE_THRESHOLDS:
        if pct >= threshold:
            return label
    return "Poor"


def _status_emoji(passed):
    return "✅" if passed else "❌"


def _bar(pct, width=30):
    filled = round(min(max(pct, 0), 100) / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _check_dependencies():
    missing = []
    for pkg in ["pytest", "pytest_cov"]:
        r = _run([sys.executable, "-c", f"import {pkg}"])
        if r.returncode != 0:
            missing.append(pkg.replace("_", "-"))
    return missing


# ---------------------------------------------------------------------------
# JUnit XML parsing — reliable, ANSI-free
# ---------------------------------------------------------------------------

def _parse_junit(xml_path: Path) -> dict:
    """Parse pytest JUnit XML into structured results."""
    results = {
        "tests": [],
        "passed": 0, "failed": 0, "error": 0, "skipped": 0,
        "duration": 0.0, "failures": [],
    }
    if not xml_path.exists():
        return results

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Root may be <testsuites> or <testsuite>
        suites = root.findall("testsuite") if root.tag == "testsuites" else [root]

        for suite in suites:
            results["duration"] += float(suite.get("time", 0))
            for case in suite.findall("testcase"):
                classname = case.get("classname", "")
                name = case.get("name", "")
                # Reconstruct module path from classname (e.g. tests.test_valuation)
                module_path = classname.split(".")[0].replace(".", "/") if "." in classname else ""
                # Convert e.g. "tests/test_valuation" -> "tests/test_valuation.py"
                if module_path and not module_path.endswith(".py"):
                    module_path = module_path + ".py"

                # Determine status
                if case.find("failure") is not None:
                    status = "FAILED"
                    fail_el = case.find("failure")
                    results["failures"].append(f"{classname}::{name}")
                    results["failed"] += 1
                elif case.find("error") is not None:
                    status = "ERROR"
                    results["error"] += 1
                    results["failures"].append(f"{classname}::{name}")
                elif case.find("skipped") is not None:
                    status = "SKIPPED"
                    results["skipped"] += 1
                else:
                    status = "PASSED"
                    results["passed"] += 1

                results["tests"].append({
                    "module": module_path,
                    "classname": classname,
                    "name": name,
                    "status": status,
                })
    except Exception as e:
        print(f"  ⚠️  Could not parse JUnit XML: {e}")

    return results


def _parse_coverage(json_path: Path) -> dict:
    """Parse coverage JSON export."""
    if not json_path.exists():
        return {"total": 0.0, "files": {}}
    try:
        data = json.loads(json_path.read_text())
        total = data.get("totals", {}).get("percent_covered", 0.0)
        files = {}
        for k, v in data.get("files", {}).items():
            # Skip test files and non-.py files
            if "tests/" in k or not k.endswith(".py"):
                continue
            pct = v.get("summary", {}).get("percent_covered", 0.0)
            files[k] = pct
        return {"total": total, "files": files}
    except Exception:
        return {"total": 0.0, "files": {}}


def _per_module_stats(results: dict) -> dict:
    """Aggregate pass/fail counts per test module file."""
    stats = {}
    for t in results["tests"]:
        # Map classname like "tests.test_valuation.TestCAPM" -> "tests/test_valuation.py"
        classname = t.get("classname", "")
        parts = classname.split(".")
        if len(parts) >= 2:
            mod_key = f"{parts[0]}/{parts[1]}.py"
        else:
            mod_key = t.get("module", "unknown")

        if mod_key not in stats:
            stats[mod_key] = {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "total": 0}
        stats[mod_key][t["status"].lower()] += 1
        stats[mod_key]["total"] += 1
    return stats


def _parse_duration_from_stdout(stdout: str) -> float:
    """Fallback: extract duration from pytest summary line."""
    m = re.search(r'in ([\d.]+)s', _strip_ansi(stdout))
    return float(m.group(1)) if m else 0.0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(results, coverage, module_runs, args, report_path):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = results["passed"] + results["failed"] + results["error"]
    pass_pct = (results["passed"] / total * 100) if total > 0 else 0.0
    overall_passed = results["failed"] == 0 and results["error"] == 0
    grade = _grade(pass_pct)
    cov_total = coverage.get("total", 0.0)
    cov_label = _cov_label(cov_total)
    per_mod = _per_module_stats(results)

    lines = []

    # ── Header ───────────────────────────────────────────────────────────────
    lines += [
        "# QuantLab — QA Report",
        "",
        f"> Generated: **{now}**  |  "
        f"Grade: **{grade}**  |  "
        f"Coverage: **{cov_total:.1f}% ({cov_label})**",
        "",
    ]

    # ── Overall Banner ────────────────────────────────────────────────────────
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

    # ── Coverage Summary ──────────────────────────────────────────────────────
    lines += [
        "## 📊 Coverage Summary",
        "",
        f"**Overall: {cov_total:.1f}%** `{_bar(cov_total)}` — {cov_label}",
        "",
        "> **Note:** Coverage reflects only the pure-Python logic paths exercised",
        "> by offline tests. Streamlit UI rendering, export functions, and",
        "> live-data tabs require an interactive session and are excluded by design.",
        "",
    ]
    if coverage.get("files"):
        lines += [
            "| File | Coverage |",
            "|------|----------|",
        ]
        for fpath, pct in sorted(coverage["files"].items(),
                                  key=lambda x: x[1], reverse=True):
            icon = "🟢" if pct >= 50 else ("🟡" if pct >= 18 else "🔴")
            lines.append(f"| `{fpath}` | {icon} {pct:.1f}% `{_bar(pct, 20)}` |")
        lines.append("")

    # ── Module Summary Table ──────────────────────────────────────────────────
    lines += [
        "## 🧪 Test Modules",
        "",
        "| Module | Description | Tests | Passed | Failed | Status |",
        "|--------|-------------|-------|--------|--------|--------|",
    ]
    for key, path in TEST_MODULES.items():
        desc = MODULE_DESCRIPTIONS.get(key, "")
        mod_stats = per_mod.get(path, {"passed": 0, "failed": 0, "error": 0, "total": 0})
        p = mod_stats["passed"]
        f = mod_stats["failed"] + mod_stats.get("error", 0)
        t = mod_stats["total"]
        rc = module_runs.get(key, (0,))[0]
        status = _status_emoji(f == 0 and rc == 0)
        lines.append(f"| `{path}` | {desc} | {t} | {p} | {f} | {status} |")
    lines.append("")

    # ── Detailed Results Per Module ───────────────────────────────────────────
    lines += ["## 📋 Detailed Test Results", ""]
    for key, path in TEST_MODULES.items():
        desc = MODULE_DESCRIPTIONS[key]
        rc = module_runs.get(key, (0,))[0]
        mod_tests = [t for t in results["tests"]
                     if t.get("classname", "").startswith(
                         path.replace("/", ".").replace(".py", ""))]
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

    # ── Failures ─────────────────────────────────────────────────────────────
    if results["failures"]:
        lines += [
            "## ❌ Failure Details",
            "",
            "| # | Failed Test |",
            "|---|------------|",
        ]
        for i, f in enumerate(results["failures"], 1):
            lines.append(f"| {i} | `{f}` |")
        lines += [
            "",
            "> Run `pytest tests/ -v --tb=long` locally for full tracebacks.",
            "",
        ]

    # ── How to Run ────────────────────────────────────────────────────────────
    lines += [
        "## 🚀 Running the Tests",
        "",
        "### Quick start",
        "```bash",
        "pip install -r requirements.txt && pip install pytest pytest-cov",
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
        "### Makefile shortcuts",
        "```bash",
        "make qa            # full suite + QA-REPORT.md",
        "make test          # pytest verbose",
        "make fast          # skip integration",
        "make coverage      # HTML coverage + auto-open",
        "make t-portfolio   # run only portfolio tests",
        "make lint          # flake8",
        "make clean         # remove test artifacts",
        "```",
        "",
    ]

    # ── Architecture ──────────────────────────────────────────────────────────
    lines += [
        "## 🏗️ Test Architecture",
        "",
        "```",
        "tests/",
        "├── conftest.py            # Streamlit stub + shared synthetic fixtures",
        "│                          # (2-year OHLCV data — fully offline, no network)",
        "├── test_valuation.py      # CAPM, Beta (ddof=1), WACC, DCF, FF, APT",
        "├── test_portfolio.py      # 9 strategies × 3 invariants (parametrized)",
        "│                          # Risk Parity, HRP, bubble-aware penalty",
        "├── test_options.py        # B-S known value, put-call parity, all 5 Greeks",
        "├── test_bubble_ml.py      # BubbleDetector, GPH SE, RSI Wilder's EMA,",
        "│                          # MACD histogram, ML pipeline, sentiment",
        "├── test_risk_and_errors.py # Risk score, exception hierarchy, ticker parser",
        "└── test_integration.py    # End-to-end: prices → portfolio → ML → options",
        "```",
        "",
        "All tests run **fully offline** — no Yahoo Finance calls, no Streamlit server.",
        "",
    ]

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        f"*Generated by `run_tests.py` · {now}*",
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
    parser.add_argument("--fast", action="store_true",
                        help="Skip integration tests.")
    parser.add_argument("--module", metavar="NAME", default=None,
                        choices=list(TEST_MODULES.keys()))
    parser.add_argument("--out", metavar="PATH", default=str(REPORT_PATH))
    parser.add_argument("--no-cov", action="store_true",
                        help="Disable coverage collection.")
    parser.add_argument("--no-report", action="store_true",
                        help="Run tests but do not write a report file.")
    args = parser.parse_args()
    report_path = Path(args.out)

    print("\n" + "═" * 60)
    print("  QuantLab QA Orchestrator")
    print("═" * 60)
    print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Report  : {report_path}")
    print(f"  Mode    : {'fast (no integration)' if args.fast else 'full suite'}")
    print("═" * 60 + "\n")

    # ── Dependency check ──────────────────────────────────────────────────────
    missing = _check_dependencies()
    if missing:
        print(f"⚠️  Missing: {', '.join(missing)}")
        answer = input("Install now? [y/N] ").strip().lower()
        if answer == "y":
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing, check=True)
        else:
            sys.exit(1)

    # ── Determine modules ─────────────────────────────────────────────────────
    if args.module:
        modules_to_run = {args.module: TEST_MODULES[args.module]}
    elif args.fast:
        modules_to_run = {k: v for k, v in TEST_MODULES.items() if k != "integration"}
    else:
        modules_to_run = TEST_MODULES

    # ── Per-module runs (for console status display) ───────────────────────────
    module_runs = {}
    print("Running test modules:")
    for key, path in modules_to_run.items():
        desc = MODULE_DESCRIPTIONS[key]
        print(f"  ▶  {desc}", end=" ... ", flush=True)
        r = _run([
            sys.executable, "-m", "pytest", path,
            "-v", "--tb=short", "--no-header", "--no-cov",
            f"--junit-xml={TESTS_DIR / f'.junit_{key}.xml'}",
        ])
        # Parse from per-module XML for accurate counts
        sub_results = _parse_junit(TESTS_DIR / f".junit_{key}.xml")
        p = sub_results["passed"]
        f = sub_results["failed"] + sub_results["error"]
        icon = "✅" if r.returncode == 0 else "❌"
        print(f"{icon}  {p} passed, {f} failed")
        module_runs[key] = (r.returncode, r.stdout, r.stderr)

    # ── Full combined run ──────────────────────────────────────────────────────
    print("\nRunning full suite for totals and coverage...")
    paths = list(modules_to_run.values())
    cmd = [
        sys.executable, "-m", "pytest", *paths,
        "-v", "--tb=short", "--no-header",
        f"--junit-xml={JUNIT_XML}",
    ]
    if not args.no_cov:
        cmd += [
            f"--cov={REPO_ROOT}",
            f"--cov-report=json:{COV_JSON}",
            "--cov-report=html:tests/coverage_html",
            "--cov-config=.coveragerc",
        ]
    else:
        cmd.append("--no-cov")

    full_run = _run(cmd)
    results = _parse_junit(JUNIT_XML)

    # Fallback: get duration from stdout if XML didn't capture it
    if results["duration"] == 0.0:
        results["duration"] = _parse_duration_from_stdout(full_run.stdout)

    coverage = _parse_coverage(COV_JSON) if not args.no_cov else {"total": 0.0, "files": {}}

    # ── Console summary ────────────────────────────────────────────────────────
    total = results["passed"] + results["failed"] + results["error"]
    pass_pct = (results["passed"] / total * 100) if total > 0 else 0.0
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

    # ── Write report ──────────────────────────────────────────────────────────
    if not args.no_report:
        _generate_report(results, coverage, module_runs, args, report_path)
        print(f"\n📄 Report written → {report_path}")
        if not args.no_cov:
            print(f"📊 HTML coverage  → tests/coverage_html/index.html")

    overall_ok = results["failed"] == 0 and results["error"] == 0
    if not overall_ok:
        print("\n💡 Run `pytest tests/ -v --tb=long` for full failure details.")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
