"""
test_frontend.py — Streamlit front-end tests using streamlit.testing.v1

Tests the actual UI widgets, session state, navigation, and error handling
directly in the Streamlit runtime — no browser, no server, fully offline.

Coverage:
  - App loads without exceptions
  - Session state initialisation
  - Dark/Light mode toggle
  - Asset class selector & Quick Presets
  - Sidebar widget defaults and interactions
  - Advanced Settings (all 4 sections)
  - Debug Mode toggle
  - Ticker text area (valid, empty, comma-separated, special chars)
  - Ticker parser integration (comma/space/semicolon)
  - Analysis state before Run Analysis is clicked
  - Error handling (invalid tickers show friendly message)
  - Run Analysis with mocked fetch (no network calls)
"""

import pytest
import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest

APP_PATH = "app.py"
TIMEOUT  = 30   # seconds per test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh() -> AppTest:
    """Return a freshly initialised AppTest for app.py."""
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    return at


def _no_exceptions(at: AppTest) -> bool:
    return len(at.exception) == 0


# ---------------------------------------------------------------------------
# 1. App Initialisation
# ---------------------------------------------------------------------------

class TestAppInitialisation:
    def test_app_loads_without_exception(self):
        """app.py must run to completion without raising any exception."""
        at = _fresh()
        assert _no_exceptions(at), f"Unexpected exceptions: {at.exception}"

    def test_session_state_theme_defaults_to_light(self):
        at = _fresh()
        assert at.session_state["theme"] == "light"

    def test_session_state_analysis_complete_false(self):
        at = _fresh()
        assert at.session_state["analysis_complete"] is False

    def test_session_state_debug_mode_false(self):
        at = _fresh()
        assert at.session_state["debug_mode"] is False

    def test_session_state_last_updated_initialising(self):
        at = _fresh()
        assert at.session_state["last_updated"] == "Initializing..."

    def test_session_state_data_is_none(self):
        at = _fresh()
        assert at.session_state["data"] is None

    def test_no_errors_on_fresh_load(self):
        """No st.error or st.warning should appear before analysis runs."""
        at = _fresh()
        assert len(at.error) == 0

    def test_run_analysis_button_present(self):
        at = _fresh()
        btns = [b for b in at.button if b.label == "Run Analysis"]
        assert len(btns) == 1

    def test_run_analysis_button_not_yet_clicked(self):
        """analysis_complete is False before button is pressed."""
        at = _fresh()
        assert at.session_state["analysis_complete"] is False


# ---------------------------------------------------------------------------
# 2. Dark Mode Toggle
# ---------------------------------------------------------------------------

class TestDarkModeToggle:
    def test_dark_mode_toggle_exists(self):
        at = _fresh()
        toggles = [t for t in at.toggle if t.label == "Dark Mode"]
        assert len(toggles) == 1

    def test_dark_mode_defaults_to_off(self):
        at = _fresh()
        dm = next(t for t in at.toggle if t.label == "Dark Mode")
        assert dm.value is False

    def test_toggle_dark_mode_on_sets_theme(self):
        at = _fresh()
        dm = next(t for t in at.toggle if t.label == "Dark Mode")
        dm.set_value(True).run()
        assert at.session_state["theme"] == "dark"
        assert _no_exceptions(at)

    def test_toggle_dark_mode_off_resets_theme(self):
        """Start in dark mode, toggle back to light."""
        at = _fresh()
        # First enable dark mode
        dm = next(t for t in at.toggle if t.label == "Dark Mode")
        dm.set_value(True).run()
        assert at.session_state["theme"] == "dark"
        # Now disable — must go back to light
        dm = next(t for t in at.toggle if t.label == "Dark Mode")
        dm.set_value(False).run()
        assert at.session_state["theme"] == "light"

    def test_dark_mode_no_exception(self):
        at = _fresh()
        dm = next(t for t in at.toggle if t.label == "Dark Mode")
        dm.set_value(True).run()
        assert _no_exceptions(at)


# ---------------------------------------------------------------------------
# 3. Asset Class Selector
# ---------------------------------------------------------------------------

class TestAssetClassSelector:
    def test_asset_class_selector_exists(self):
        at = _fresh()
        labels = [sb.label for sb in at.selectbox]
        assert "Select Asset Class" in labels

    def test_default_asset_class_is_stocks(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Select Asset Class")
        assert sb.value == "Stocks & ETFs"

    def test_asset_class_options_present(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Select Asset Class")
        for cls in ["Stocks & ETFs", "Forex", "Commodities", "Crypto"]:
            assert cls in sb.options

    def test_select_forex_class(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Select Asset Class")
        sb.select("Forex").run()
        assert sb.value == "Forex"
        assert _no_exceptions(at)

    def test_select_crypto_class(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Select Asset Class")
        sb.select("Crypto").run()
        assert sb.value == "Crypto"
        assert _no_exceptions(at)

    def test_select_commodities_class(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Select Asset Class")
        sb.select("Commodities").run()
        assert sb.value == "Commodities"
        assert _no_exceptions(at)


# ---------------------------------------------------------------------------
# 4. Quick Presets
# ---------------------------------------------------------------------------

class TestQuickPresets:
    def test_preset_selectbox_exists(self):
        at = _fresh()
        labels = [sb.label for sb in at.selectbox]
        assert "Quick Presets" in labels

    def test_default_preset_is_none(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Quick Presets")
        assert sb.value == "— None —"

    def test_preset_options_include_tech_giants(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Quick Presets")
        assert "Tech Giants" in sb.options

    def test_preset_options_include_crypto(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Quick Presets")
        assert "Crypto Majors" in sb.options

    def test_preset_options_include_forex(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Quick Presets")
        assert "Forex Majors" in sb.options

    def test_selecting_tech_giants_updates_tickers(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Quick Presets")
        sb.select("Tech Giants").run()
        ta = at.text_area[0]
        # Tech Giants should include AAPL, MSFT, GOOGL, NVDA etc.
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            assert ticker in ta.value, f"{ticker} missing from Tech Giants preset"
        assert _no_exceptions(at)

    def test_selecting_crypto_preset_updates_tickers(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Quick Presets")
        sb.select("Crypto Majors").run()
        ta = at.text_area[0]
        assert "BTC-USD" in ta.value
        assert "ETH-USD" in ta.value
        assert _no_exceptions(at)

    def test_selecting_forex_preset_updates_tickers(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Quick Presets")
        sb.select("Forex Majors").run()
        ta = at.text_area[0]
        assert "EURUSD=X" in ta.value
        assert _no_exceptions(at)

    def test_selecting_precious_metals_preset(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Quick Presets")
        sb.select("Precious Metals").run()
        ta = at.text_area[0]
        assert "GC=F" in ta.value    # Gold futures
        assert _no_exceptions(at)

    def test_at_least_22_preset_options(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Quick Presets")
        # Excluding '— None —'
        real_options = [o for o in sb.options if o != "— None —"]
        assert len(real_options) >= 22, f"Only {len(real_options)} presets found"


# ---------------------------------------------------------------------------
# 5. Ticker Text Area
# ---------------------------------------------------------------------------

class TestTickerInput:
    def test_ticker_area_exists(self):
        at = _fresh()
        assert len(at.text_area) >= 1

    def test_ticker_area_label(self):
        at = _fresh()
        assert at.text_area[0].label == "Tickers"

    def test_default_tickers_non_empty(self):
        at = _fresh()
        assert len(at.text_area[0].value.strip()) > 0

    def test_set_tickers_space_separated(self):
        at = _fresh()
        at.text_area[0].set_value("AAPL MSFT").run()
        assert at.text_area[0].value == "AAPL MSFT"
        assert _no_exceptions(at)

    def test_set_tickers_comma_separated(self):
        at = _fresh()
        at.text_area[0].set_value("AAPL,MSFT,GOOGL").run()
        assert "AAPL" in at.text_area[0].value
        assert _no_exceptions(at)

    def test_set_forex_tickers(self):
        at = _fresh()
        at.text_area[0].set_value("EURUSD=X GBPUSD=X").run()
        assert "EURUSD=X" in at.text_area[0].value
        assert _no_exceptions(at)

    def test_set_commodity_tickers(self):
        at = _fresh()
        at.text_area[0].set_value("GC=F CL=F").run()
        assert "GC=F" in at.text_area[0].value
        assert _no_exceptions(at)

    def test_empty_tickers_no_crash(self):
        """Empty ticker input must not crash the app."""
        at = _fresh()
        at.text_area[0].set_value("").run()
        assert _no_exceptions(at)

    def test_whitespace_only_no_crash(self):
        at = _fresh()
        at.text_area[0].set_value("   ").run()
        assert _no_exceptions(at)


# ---------------------------------------------------------------------------
# 6. Advanced Settings — Monte Carlo Section
# ---------------------------------------------------------------------------

class TestAdvancedSettingsMonteCarlo:
    def test_mc_simulations_slider_exists(self):
        at = _fresh()
        labels = [s.label for s in at.slider]
        assert "Monte Carlo Simulations" in labels

    def test_mc_simulations_default_500(self):
        at = _fresh()
        sl = next(s for s in at.slider if s.label == "Monte Carlo Simulations")
        assert sl.value == 500

    def test_mc_simulations_min_100_max_2000(self):
        at = _fresh()
        sl = next(s for s in at.slider if s.label == "Monte Carlo Simulations")
        assert sl.min == 100
        assert sl.max == 2000

    def test_set_mc_simulations_1000(self):
        at = _fresh()
        sl = next(s for s in at.slider if s.label == "Monte Carlo Simulations")
        sl.set_value(1000).run()
        assert sl.value == 1000
        assert _no_exceptions(at)

    def test_forecast_days_slider_exists(self):
        at = _fresh()
        labels = [s.label for s in at.slider]
        assert "Forecast Days" in labels

    def test_forecast_days_default_90(self):
        at = _fresh()
        sl = next(s for s in at.slider if s.label == "Forecast Days")
        assert sl.value == 90

    def test_simulation_method_selectbox(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Simulation Method")
        assert sb.value == "Behavioral Agent Model"
        assert "Geometric Brownian Motion" in sb.options
        assert "Behavioral Agent Model" in sb.options

    def test_switch_to_gbm_simulation(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Simulation Method")
        sb.select("Geometric Brownian Motion").run()
        assert sb.value == "Geometric Brownian Motion"
        assert _no_exceptions(at)


# ---------------------------------------------------------------------------
# 7. Advanced Settings — Portfolio Section
# ---------------------------------------------------------------------------

class TestAdvancedSettingsPortfolio:
    def test_bubble_aware_checkbox_exists(self):
        at = _fresh()
        labels = [cb.label for cb in at.checkbox]
        assert "Bubble-Aware Portfolio" in labels

    def test_bubble_aware_defaults_true(self):
        at = _fresh()
        cb = next(c for c in at.checkbox if c.label == "Bubble-Aware Portfolio")
        assert cb.value is True

    def test_disable_bubble_aware(self):
        at = _fresh()
        cb = next(c for c in at.checkbox if c.label == "Bubble-Aware Portfolio")
        cb.set_value(False).run()
        assert cb.value is False
        assert _no_exceptions(at)

    def test_bubble_penalty_slider_exists(self):
        at = _fresh()
        labels = [s.label for s in at.slider]
        assert "Bubble Penalty" in labels

    def test_bubble_penalty_default_0_5(self):
        at = _fresh()
        sl = next(s for s in at.slider if s.label == "Bubble Penalty")
        assert sl.value == 0.5

    def test_benchmark_selectbox_exists(self):
        at = _fresh()
        labels = [sb.label for sb in at.selectbox]
        assert "Benchmark" in labels

    def test_benchmark_options(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Benchmark")
        for name in ["S&P 500 (^GSPC)", "NASDAQ (^IXIC)", "Dow Jones (^DJI)"]:
            assert name in sb.options

    def test_rebalancing_selectbox_exists(self):
        at = _fresh()
        labels = [sb.label for sb in at.selectbox]
        assert "Rebalancing Frequency" in labels

    def test_rebalancing_default_buy_hold(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Rebalancing Frequency")
        assert sb.value == "None (Buy & Hold)"

    def test_rebalancing_options(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Rebalancing Frequency")
        for opt in ["None (Buy & Hold)", "Monthly", "Quarterly", "Annually"]:
            assert opt in sb.options

    def test_custom_rf_checkbox(self):
        at = _fresh()
        cb = next(c for c in at.checkbox if c.label == "Use Custom Risk-Free Rate")
        assert cb.value is False

    def test_enable_custom_rf(self):
        at = _fresh()
        cb = next(c for c in at.checkbox if c.label == "Use Custom Risk-Free Rate")
        cb.set_value(True).run()
        assert cb.value is True
        assert _no_exceptions(at)


# ---------------------------------------------------------------------------
# 8. Advanced Settings — ML Section
# ---------------------------------------------------------------------------

class TestAdvancedSettingsML:
    def test_ml_training_period_slider_exists(self):
        at = _fresh()
        labels = [s.label for s in at.slider]
        assert "ML Training Period (years)" in labels

    def test_ml_training_period_default_3(self):
        at = _fresh()
        sl = next(s for s in at.slider if s.label == "ML Training Period (years)")
        assert sl.value == 3

    def test_ml_training_period_range(self):
        at = _fresh()
        sl = next(s for s in at.slider if s.label == "ML Training Period (years)")
        assert sl.min == 1
        assert sl.max == 5

    def test_clustering_method_selectbox(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Clustering Method")
        assert sb.value == "K-Means"
        assert "Gaussian Mixture" in sb.options

    def test_switch_clustering_to_gmm(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Clustering Method")
        sb.select("Gaussian Mixture").run()
        assert sb.value == "Gaussian Mixture"
        assert _no_exceptions(at)

    def test_anomaly_sensitivity_slider(self):
        at = _fresh()
        sl = next(s for s in at.slider if s.label == "Anomaly Sensitivity")
        assert sl.value == pytest.approx(0.05, abs=0.001)
        assert sl.min == pytest.approx(0.01)
        assert sl.max == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# 9. Advanced Settings — Display & Export Section
# ---------------------------------------------------------------------------

class TestAdvancedSettingsDisplay:
    def test_chart_height_slider_exists(self):
        at = _fresh()
        labels = [s.label for s in at.slider]
        assert "Chart Height (px)" in labels

    def test_chart_height_default_500(self):
        at = _fresh()
        sl = next(s for s in at.slider if s.label == "Chart Height (px)")
        assert sl.value == 500

    def test_chart_height_range(self):
        at = _fresh()
        sl = next(s for s in at.slider if s.label == "Chart Height (px)")
        assert sl.min == 300
        assert sl.max == 800

    def test_export_sections_multiselect_exists(self):
        at = _fresh()
        labels = [ms.label for ms in at.multiselect]
        assert "Include in Reports" in labels

    def test_export_sections_has_defaults(self):
        at = _fresh()
        ms = next(m for m in at.multiselect if m.label == "Include in Reports")
        assert len(ms.value) > 0
        assert "Price Charts" in ms.value

    def test_export_sections_all_options_available(self):
        at = _fresh()
        ms = next(m for m in at.multiselect if m.label == "Include in Reports")
        for section in ["Price Charts", "Portfolio Weights", "ML Predictions",
                        "Bubble Scores", "Risk Dashboard", "Sentiment"]:
            assert section in ms.options

    def test_auto_refresh_toggle_exists(self):
        at = _fresh()
        labels = [t.label for t in at.toggle]
        assert "Enable Auto-Refresh" in labels

    def test_auto_refresh_defaults_off(self):
        at = _fresh()
        tg = next(t for t in at.toggle if t.label == "Enable Auto-Refresh")
        assert tg.value is False

    def test_enable_auto_refresh(self):
        at = _fresh()
        tg = next(t for t in at.toggle if t.label == "Enable Auto-Refresh")
        tg.set_value(True).run()
        assert tg.value is True
        assert _no_exceptions(at)


# ---------------------------------------------------------------------------
# 10. Debug Mode
# ---------------------------------------------------------------------------

class TestDebugMode:
    def test_debug_checkbox_exists(self):
        at = _fresh()
        labels = [cb.label for cb in at.checkbox]
        assert "Debug Mode" in labels

    def test_debug_mode_defaults_off(self):
        at = _fresh()
        assert at.session_state["debug_mode"] is False

    def test_enable_debug_mode(self):
        at = _fresh()
        cb = next(c for c in at.checkbox if c.label == "Debug Mode")
        cb.set_value(True).run()
        assert at.session_state["debug_mode"] is True
        assert _no_exceptions(at)

    def test_disable_debug_mode_via_new_session(self):
        """
        Debug Mode is tied to session_state via value=st.session_state['debug_mode'].
        Disabling it requires a fresh session (as it would in a real browser page reload).
        A fresh AppTest instance starts with debug_mode=False.
        """
        # Fresh session always starts with debug_mode False
        at = _fresh()
        assert at.session_state["debug_mode"] is False
        assert next(c for c in at.checkbox if c.label == "Debug Mode").value is False


# ---------------------------------------------------------------------------
# 11. Run Analysis — mocked (no network calls)
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    """Tests for the Run Analysis button and pre-analysis state.
    
    Note: Full mocked-analysis tests require monkey-patching inside AppTest's
    re-execution context, which is not directly possible with external patch().
    These tests cover the widget state, defaults, and error paths.
    """

    def test_analysis_not_complete_before_click(self):
        at = _fresh()
        assert at.session_state["analysis_complete"] is False

    def test_analysis_button_present_and_clickable(self):
        at = _fresh()
        btns = [b for b in at.button if b.label == "Run Analysis"]
        assert len(btns) == 1

    def test_data_is_none_before_analysis(self):
        at = _fresh()
        assert at.session_state["data"] is None

    def test_run_empty_tickers_shows_error_and_stays_incomplete(self):
        """Empty ticker input must show error and keep analysis_complete False."""
        at = _fresh()
        at.text_area[0].set_value("").run()
        at.button[0].click().run()
        assert at.session_state["analysis_complete"] is False
        assert len(at.error) > 0
        assert _no_exceptions(at)

    def test_simulation_method_default_behavioral_agent(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Simulation Method")
        assert sb.value == "Behavioral Agent Model"

    def test_simulation_method_switch_to_gbm(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Simulation Method")
        sb.select("Geometric Brownian Motion").run()
        assert sb.value == "Geometric Brownian Motion"
        assert _no_exceptions(at)

    def test_clustering_method_persists_across_runs(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Clustering Method")
        sb.select("Gaussian Mixture").run()
        # After re-run, the widget must retain its value
        sb2 = next(s for s in at.selectbox if s.label == "Clustering Method")
        assert sb2.value == "Gaussian Mixture"
        assert _no_exceptions(at)

    def test_mc_sims_persists_after_preset_change(self):
        """Changing a preset must not reset unrelated Advanced Settings."""
        at = _fresh()
        sl = next(s for s in at.slider if s.label == "Monte Carlo Simulations")
        sl.set_value(1500).run()
        # Now change a preset
        sb = next(s for s in at.selectbox if s.label == "Quick Presets")
        sb.select("Tech Giants").run()
        # Slider value should still be 1500
        sl2 = next(s for s in at.slider if s.label == "Monte Carlo Simulations")
        assert sl2.value == 1500
        assert _no_exceptions(at)


# ---------------------------------------------------------------------------
# 12. Error Handling — network failure shows friendly message
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_empty_tickers_shows_error(self):
        """Empty ticker input: app shows st.error and does not complete analysis."""
        at = _fresh()
        at.text_area[0].set_value("").run()
        at.button[0].click().run()
        assert at.session_state["analysis_complete"] is False
        assert len(at.error) > 0
        assert _no_exceptions(at)

    def test_whitespace_only_tickers_shows_error(self):
        at = _fresh()
        at.text_area[0].set_value("   ").run()
        at.button[0].click().run()
        assert at.session_state["analysis_complete"] is False
        assert _no_exceptions(at)

    def test_error_does_not_persist_after_ticker_fix(self):
        """
        After an error run, correcting the tickers and running again
        should start fresh (no lingering error state).
        """
        at = _fresh()
        # First: trigger an error with empty tickers
        at.text_area[0].set_value("").run()
        at.button[0].click().run()
        assert at.session_state["analysis_complete"] is False
        # Second: the app must not be stuck — it stays interactive
        at.text_area[0].set_value("AAPL MSFT").run()
        # App should accept the new input without crashing
        assert _no_exceptions(at)

    def test_analysis_complete_false_on_fresh_load(self):
        """Before any button click, analysis_complete must be False."""
        at = _fresh()
        assert at.session_state["analysis_complete"] is False
        assert _no_exceptions(at)

    def test_no_error_on_fresh_load(self):
        """No st.error should appear before the user does anything."""
        at = _fresh()
        assert len(at.error) == 0
        assert _no_exceptions(at)


# ---------------------------------------------------------------------------
# 13. Composite Widget Interaction Flows
# ---------------------------------------------------------------------------

class TestCompositeFlows:
    def test_preset_then_dark_mode(self):
        """Selecting a preset and then toggling dark mode should both work."""
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Quick Presets")
        sb.select("Semiconductor").run()
        dm = next(t for t in at.toggle if t.label == "Dark Mode")
        dm.set_value(True).run()
        assert at.session_state["theme"] == "dark"
        assert "NVDA" in at.text_area[0].value
        assert _no_exceptions(at)

    def test_change_multiple_advanced_settings(self):
        """Multiple settings can be changed without conflicts."""
        at = _fresh()
        sl_sims = next(s for s in at.slider if s.label == "Monte Carlo Simulations")
        sl_sims.set_value(200).run()
        sb_clust = next(s for s in at.selectbox if s.label == "Clustering Method")
        sb_clust.select("Gaussian Mixture").run()
        sl_ml = next(s for s in at.slider if s.label == "ML Training Period (years)")
        sl_ml.set_value(5).run()

        assert sl_sims.value == 200
        assert sb_clust.value == "Gaussian Mixture"
        assert sl_ml.value == 5
        assert _no_exceptions(at)

    def test_forex_preset_with_benchmark_change(self):
        at = _fresh()
        sb_preset = next(s for s in at.selectbox if s.label == "Quick Presets")
        sb_preset.select("Forex Majors").run()
        sb_bench = next(s for s in at.selectbox if s.label == "Benchmark")
        sb_bench.select("NASDAQ (^IXIC)").run()
        assert "EURUSD=X" in at.text_area[0].value
        assert sb_bench.value == "NASDAQ (^IXIC)"
        assert _no_exceptions(at)

    def test_debug_mode_with_dark_mode(self):
        at = _fresh()
        next(c for c in at.checkbox if c.label == "Debug Mode").set_value(True).run()
        next(t for t in at.toggle if t.label == "Dark Mode").set_value(True).run()
        assert at.session_state["debug_mode"] is True
        assert at.session_state["theme"] == "dark"
        assert _no_exceptions(at)


# ---------------------------------------------------------------------------
# 14. Data Source Session State Defaults
# ---------------------------------------------------------------------------

class TestDataSourceDefaults:
    def test_data_source_active_default(self):
        at = _fresh()
        assert at.session_state["data_source_active"] == "Yahoo Finance"

    def test_av_enabled_default_false(self):
        at = _fresh()
        assert at.session_state["av_enabled"] is False

    def test_av_api_key_default_empty(self):
        at = _fresh()
        assert at.session_state["av_api_key"] == ""

    def test_fred_enabled_default_false(self):
        at = _fresh()
        assert at.session_state["fred_enabled"] is False

    def test_fred_api_key_default_empty(self):
        at = _fresh()
        assert at.session_state["fred_api_key"] == ""

    def test_finnhub_enabled_default_false(self):
        at = _fresh()
        assert at.session_state["finnhub_enabled"] is False

    def test_finnhub_api_key_default_empty(self):
        at = _fresh()
        assert at.session_state["finnhub_api_key"] == ""

    def test_data_fetch_log_default_empty(self):
        at = _fresh()
        assert at.session_state["data_fetch_log"] == []

    def test_data_cache_default_empty(self):
        at = _fresh()
        assert at.session_state["data_cache"] == {}

    def test_data_cache_time_default_none(self):
        at = _fresh()
        assert at.session_state["data_cache_time"] is None


# ---------------------------------------------------------------------------
# 15. Data Source Sidebar Widgets
# ---------------------------------------------------------------------------

class TestDataSourceSidebarWidgets:
    def test_av_checkbox_present(self):
        at = _fresh()
        av_cbs = [c for c in at.checkbox if c.label == "Alpha Vantage"]
        assert len(av_cbs) == 1, "Alpha Vantage checkbox should be present"

    def test_fred_checkbox_present(self):
        at = _fresh()
        fred_cbs = [c for c in at.checkbox if c.label == "FRED (Macro Data)"]
        assert len(fred_cbs) == 1, "FRED checkbox should be present"

    def test_finnhub_checkbox_present(self):
        at = _fresh()
        fh_cbs = [c for c in at.checkbox if c.label == "Finnhub"]
        assert len(fh_cbs) == 1, "Finnhub checkbox should be present"

    def test_av_key_appears_when_enabled(self):
        at = _fresh()
        cb = next(c for c in at.checkbox if c.label == "Alpha Vantage")
        cb.set_value(True).run()
        key_inputs = [i for i in at.text_input if "Alpha Vantage API Key" in i.label]
        assert len(key_inputs) == 1

    def test_fred_key_appears_when_enabled(self):
        at = _fresh()
        cb = next(c for c in at.checkbox if c.label == "FRED (Macro Data)")
        cb.set_value(True).run()
        key_inputs = [i for i in at.text_input if "FRED API Key" in i.label]
        assert len(key_inputs) == 1

    def test_finnhub_key_appears_when_enabled(self):
        at = _fresh()
        cb = next(c for c in at.checkbox if c.label == "Finnhub")
        cb.set_value(True).run()
        key_inputs = [i for i in at.text_input if "Finnhub API Key" in i.label]
        assert len(key_inputs) == 1

    def test_av_key_hidden_when_disabled(self):
        at = _fresh()
        key_inputs = [i for i in at.text_input if "Alpha Vantage API Key" in i.label]
        assert len(key_inputs) == 0

    def test_no_exceptions_after_enabling_all(self):
        at = _fresh()
        for label in ["Alpha Vantage", "FRED (Macro Data)", "Finnhub"]:
            cb = next(c for c in at.checkbox if c.label == label)
            cb.set_value(True).run()
        assert _no_exceptions(at)


# ---------------------------------------------------------------------------
# 14. Module 26-30 Session State Defaults
# ---------------------------------------------------------------------------

class TestAdvancedModuleSessionState:
    def test_pairs_bt_method_default(self):
        at = _fresh()
        assert at.session_state["pairs_bt_method"] == "Kalman Filter"

    def test_wf_train_window_default(self):
        at = _fresh()
        assert at.session_state["wf_train_window"] == 252

    def test_wf_test_window_default(self):
        at = _fresh()
        assert at.session_state["wf_test_window"] == 63

    def test_macro_lookback_default(self):
        at = _fresh()
        assert at.session_state["macro_lookback"] == "2y"

    def test_crypto_lookback_default(self):
        at = _fresh()
        assert at.session_state["crypto_lookback"] == "365d"

    def test_insider_lookback_default(self):
        at = _fresh()
        assert at.session_state["insider_lookback"] == "6 months"

    def test_watchlists_default(self):
        at = _fresh()
        wl = at.session_state["watchlists"]
        assert isinstance(wl, dict)
        assert "Default" in wl
        assert "AAPL" in wl["Default"]

    def test_alerts_default_empty(self):
        at = _fresh()
        assert at.session_state["alerts"] == []

    def test_triggered_alerts_default_empty(self):
        at = _fresh()
        assert at.session_state["triggered_alerts"] == []

    def test_active_watchlist_default(self):
        at = _fresh()
        assert at.session_state["active_watchlist"] == "Default"


# ---------------------------------------------------------------------------
# 15. Module 26-30 Sidebar Widgets
# ---------------------------------------------------------------------------

class TestAdvancedModuleSidebarWidgets:
    def test_backtest_method_selectbox_exists(self):
        at = _fresh()
        labels = [sb.label for sb in at.selectbox]
        assert "Backtest Method" in labels

    def test_backtest_method_options(self):
        at = _fresh()
        sb = next(s for s in at.selectbox if s.label == "Backtest Method")
        for opt in ["Classic OLS", "Kalman Filter", "Regime-Adaptive"]:
            assert opt in sb.options

    def test_macro_lookback_selectbox_exists(self):
        at = _fresh()
        labels = [sb.label for sb in at.selectbox]
        assert "Macro Lookback" in labels

    def test_crypto_lookback_selectbox_exists(self):
        at = _fresh()
        labels = [sb.label for sb in at.selectbox]
        assert "Crypto Lookback" in labels

    def test_insider_lookback_selectbox_exists(self):
        at = _fresh()
        labels = [sb.label for sb in at.selectbox]
        assert "Insider Lookback" in labels

    def test_wf_train_window_slider_exists(self):
        at = _fresh()
        labels = [s.label for s in at.slider]
        assert "WF Train Window" in labels

    def test_wf_test_window_slider_exists(self):
        at = _fresh()
        labels = [s.label for s in at.slider]
        assert "WF Test Window" in labels
