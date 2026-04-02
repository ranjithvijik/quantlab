# ============================================================
#  QuantLab — Makefile
#  Usage: make <target>
# ============================================================

PYTHON   := python3
PIP      := $(PYTHON) -m pip
PYTEST   := $(PYTHON) -m pytest
RUNNER   := $(PYTHON) run_tests.py
TESTS    := tests/
REPORT   := QA-REPORT.md
COV_HTML := tests/coverage_html/index.html

.DEFAULT_GOAL := help

# ── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@echo "  QuantLab — available make targets"
	@echo "  ──────────────────────────────────"
	@echo "  make install    Install all Python dependencies"
	@echo "  make test       Run full test suite (verbose)"
	@echo "  make qa         Run full suite + write QA-REPORT.md"
	@echo "  make fast       Run unit tests only (skip integration)"
	@echo "  make coverage   Run suite with HTML coverage, then open report"
	@echo "  make lint       Run flake8 + basic checks"
	@echo "  make clean      Remove generated test artifacts"
	@echo "  make report     Re-open the last QA-REPORT.md (no re-run)"
	@echo ""
	@echo "  Per-module shortcuts:"
	@echo "  make t-valuation   make t-portfolio   make t-options"
	@echo "  make t-bubble      make t-risk        make t-integration"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────

.PHONY: install
install:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov flake8

# ── Testing ──────────────────────────────────────────────────────────────────

.PHONY: test
test:
	$(PYTEST) $(TESTS) -v --tb=short

.PHONY: qa
qa:
	$(RUNNER)
	@echo ""
	@echo "  ✅  QA complete. Report → $(REPORT)"

.PHONY: fast
fast:
	$(RUNNER) --fast
	@echo ""
	@echo "  ✅  Fast QA complete (integration skipped). Report → $(REPORT)"

.PHONY: coverage
coverage:
	$(PYTEST) $(TESTS) -v --tb=short \
		--cov=. \
		--cov-report=html:tests/coverage_html \
		--cov-report=term-missing:skip-covered \
		--cov-config=.coveragerc
	@echo ""
	@echo "  📊  Coverage report → $(COV_HTML)"
	@(which open && open $(COV_HTML)) || \
	 (which xdg-open && xdg-open $(COV_HTML)) || \
	 echo "  Open $(COV_HTML) in your browser."

.PHONY: report
report:
	@(which open && open $(REPORT)) || \
	 (which xdg-open && xdg-open $(REPORT)) || \
	 cat $(REPORT)

# ── Per-module targets ────────────────────────────────────────────────────────

.PHONY: t-valuation
t-valuation:
	$(RUNNER) --module valuation

.PHONY: t-portfolio
t-portfolio:
	$(RUNNER) --module portfolio

.PHONY: t-options
t-options:
	$(RUNNER) --module options

.PHONY: t-bubble
t-bubble:
	$(RUNNER) --module bubble_ml

.PHONY: t-risk
t-risk:
	$(RUNNER) --module risk_errors

.PHONY: t-integration
t-integration:
	$(RUNNER) --module integration

.PHONY: t-frontend
t-frontend:
	$(RUNNER) --module frontend

# ── Lint ─────────────────────────────────────────────────────────────────────

.PHONY: lint
lint:
	$(PYTHON) -m flake8 app.py --max-line-length=120 \
		--ignore=E501,W503,E203,E402 \
		--exclude=tests/ \
		|| true
	@echo "  ✅  Lint check complete."

# ── Cleanup ───────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -f $(REPORT)
	rm -f tests/.junit.xml tests/.coverage.json
	rm -f .coverage
	rm -rf tests/coverage_html
	rm -rf __pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "  🧹  Cleaned test artifacts."
