# QuantLab — QA Report

> Generated: **2026-04-02 19:41 UTC**  |  Grade: **A+**  |  Coverage: **0.0% (Poor)**

## 🟢 ALL TESTS PASSED

| Metric | Value |
|--------|-------|
| Total Tests | **274** |
| Passed | ✅ 274 |
| Failed | ✅ 0 |
| Errors | ✅ 0 |
| Skipped | ⏭️ 0 |
| Pass Rate | 100.0% `██████████████████████████████` |
| Duration | ⏱️ 61.73s |
| Grade | **A+** |

## 📊 Coverage Summary

**Overall: 0.0%** `░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` — Poor

> **Note:** Coverage reflects only the pure-Python logic paths exercised
> by offline tests. Streamlit UI rendering, export functions, and
> live-data tabs require an interactive session and are excluded by design.

## 🧪 Test Modules

| Module | Description | Tests | Passed | Failed | Status |
|--------|-------------|-------|--------|--------|--------|
| `tests/unit/test_valuation.py` | Valuation Models (CAPM, Beta, WACC, DCF, Fama-French, APT) | 19 | 19 | 0 | ✅ |
| `tests/unit/test_portfolio.py` | Portfolio Optimization (9 strategies, risk matrices, bubble-aware) | 43 | 43 | 0 | ✅ |
| `tests/unit/test_options.py` | Options Pricing (Black-Scholes, Greeks, Payoff Diagrams) | 28 | 28 | 0 | ✅ |
| `tests/unit/test_bubble_ml.py` | Bubble Detection, Technical Indicators & ML Pipeline | 37 | 37 | 0 | ✅ |
| `tests/unit/test_risk_and_errors.py` | Risk Score, Error Handling & Ticker Parser | 36 | 36 | 0 | ✅ |
| `tests/unit/test_integration.py` | End-to-End Integration Pipeline | 16 | 16 | 0 | ✅ |
| `tests/frontend/test_frontend.py` | Streamlit UI — Widgets, Presets, Session State, Run Analysis | 95 | 95 | 0 | ✅ |

## 📋 Detailed Test Results

### Valuation Models (CAPM, Beta, WACC, DCF, Fama-French, APT)
**File:** `tests/unit/test_valuation.py` &nbsp;|&nbsp; **Status:** ✅ PASSED &nbsp;|&nbsp; **19/19 tests passing**

<details>
<summary>Show all tests</summary>

| Test | Status |
|------|--------|
| `test_capm_zero_beta` | ✅ PASSED |
| `test_capm_unit_beta` | ✅ PASSED |
| `test_capm_high_beta` | ✅ PASSED |
| `test_capm_negative_beta` | ✅ PASSED |
| `test_capm_return_type` | ✅ PASSED |
| `test_beta_known_value` | ✅ PASSED |
| `test_beta_market_itself_is_one` | ✅ PASSED |
| `test_beta_insufficient_data` | ✅ PASSED |
| `test_beta_uses_sample_variance` | ✅ PASSED |
| `test_beta_non_overlapping_dates` | ✅ PASSED |
| `test_wacc_returns_float` | ✅ PASSED |
| `test_wacc_reasonable_range` | ✅ PASSED |
| `test_wacc_increases_with_beta` | ✅ PASSED |
| `test_wacc_higher_rf_raises_wacc` | ✅ PASSED |
| `test_dcf_returns_value_or_none` | ✅ PASSED |
| `test_dcf_wacc_le_terminal_growth_returns_none` | ✅ PASSED |
| `test_fama_french_returns_float` | ✅ PASSED |
| `test_fama_french_positive_expected_return` | ✅ PASSED |
| `test_apt_returns_float` | ✅ PASSED |

</details>

### Portfolio Optimization (9 strategies, risk matrices, bubble-aware)
**File:** `tests/unit/test_portfolio.py` &nbsp;|&nbsp; **Status:** ✅ PASSED &nbsp;|&nbsp; **43/43 tests passing**

<details>
<summary>Show all tests</summary>

| Test | Status |
|------|--------|
| `test_attributes` | ✅ PASSED |
| `test_cov_matrix_positive_semidefinite` | ✅ PASSED |
| `test_semi_cov_shape` | ✅ PASSED |
| `test_cvar_matrix_shape` | ✅ PASSED |
| `test_weights_sum_to_one[maximum_sharpe]` | ✅ PASSED |
| `test_weights_sum_to_one[minimum_variance]` | ✅ PASSED |
| `test_weights_sum_to_one[risk_parity]` | ✅ PASSED |
| `test_weights_sum_to_one[minimum_cvar]` | ✅ PASSED |
| `test_weights_sum_to_one[maximum_diversification]` | ✅ PASSED |
| `test_weights_sum_to_one[kelly_criterion]` | ✅ PASSED |
| `test_weights_sum_to_one[black_litterman]` | ✅ PASSED |
| `test_weights_sum_to_one[hierarchical_risk_parity]` | ✅ PASSED |
| `test_weights_sum_to_one[equal_weight]` | ✅ PASSED |
| `test_weights_non_negative[maximum_sharpe]` | ✅ PASSED |
| `test_weights_non_negative[minimum_variance]` | ✅ PASSED |
| `test_weights_non_negative[risk_parity]` | ✅ PASSED |
| `test_weights_non_negative[minimum_cvar]` | ✅ PASSED |
| `test_weights_non_negative[maximum_diversification]` | ✅ PASSED |
| `test_weights_non_negative[kelly_criterion]` | ✅ PASSED |
| `test_weights_non_negative[black_litterman]` | ✅ PASSED |
| `test_weights_non_negative[hierarchical_risk_parity]` | ✅ PASSED |
| `test_weights_non_negative[equal_weight]` | ✅ PASSED |
| `test_weights_length[maximum_sharpe]` | ✅ PASSED |
| `test_weights_length[minimum_variance]` | ✅ PASSED |
| `test_weights_length[risk_parity]` | ✅ PASSED |
| `test_weights_length[minimum_cvar]` | ✅ PASSED |
| `test_weights_length[maximum_diversification]` | ✅ PASSED |
| `test_weights_length[kelly_criterion]` | ✅ PASSED |
| `test_weights_length[black_litterman]` | ✅ PASSED |
| `test_weights_length[hierarchical_risk_parity]` | ✅ PASSED |
| `test_weights_length[equal_weight]` | ✅ PASSED |
| `test_sharpe_exceeds_min_var` | ✅ PASSED |
| `test_vol_le_equal_weight` | ✅ PASSED |
| `test_equal_weight_is_uniform` | ✅ PASSED |
| `test_risk_contributions_equal` | ✅ PASSED |
| `test_hrp_non_trivial` | ✅ PASSED |
| `test_bubble_reduces_weight` | ✅ PASSED |
| `test_metrics_keys` | ✅ PASSED |
| `test_volatility_positive` | ✅ PASSED |
| `test_sharpe_includes_rf` | ✅ PASSED |
| `test_max_drawdown_non_positive` | ✅ PASSED |
| `test_semi_cov_uses_downside_rows` | ✅ PASSED |
| `test_cvar_matrix_portfolio_level_tail` | ✅ PASSED |

</details>

### Options Pricing (Black-Scholes, Greeks, Payoff Diagrams)
**File:** `tests/unit/test_options.py` &nbsp;|&nbsp; **Status:** ✅ PASSED &nbsp;|&nbsp; **28/28 tests passing**

<details>
<summary>Show all tests</summary>

| Test | Status |
|------|--------|
| `test_call_positive` | ✅ PASSED |
| `test_put_positive` | ✅ PASSED |
| `test_put_call_parity` | ✅ PASSED |
| `test_call_intrinsic_at_expiry` | ✅ PASSED |
| `test_put_intrinsic_at_expiry` | ✅ PASSED |
| `test_deep_itm_call_approaches_forward` | ✅ PASSED |
| `test_higher_vol_higher_price` | ✅ PASSED |
| `test_longer_expiry_higher_price` | ✅ PASSED |
| `test_known_value` | ✅ PASSED |
| `test_zero_sigma_returns_intrinsic` | ✅ PASSED |
| `test_call_delta_between_0_and_1` | ✅ PASSED |
| `test_put_delta_between_minus1_and_0` | ✅ PASSED |
| `test_call_delta_atm_approx_half` | ✅ PASSED |
| `test_put_call_delta_sum_equals_one` | ✅ PASSED |
| `test_gamma_positive` | ✅ PASSED |
| `test_vega_positive` | ✅ PASSED |
| `test_call_theta_negative` | ✅ PASSED |
| `test_call_rho_positive` | ✅ PASSED |
| `test_put_rho_negative` | ✅ PASSED |
| `test_gamma_call_equals_put` | ✅ PASSED |
| `test_vega_call_equals_put` | ✅ PASSED |
| `test_expiry_returns_zeros` | ✅ PASSED |
| `test_long_call_floor` | ✅ PASSED |
| `test_long_put_floor` | ✅ PASSED |
| `test_straddle_v_shape` | ✅ PASSED |
| `test_iron_condor_limited_loss` | ✅ PASSED |
| `test_bull_call_spread_capped_profit` | ✅ PASSED |
| `test_bear_put_spread_max_gain` | ✅ PASSED |

</details>

### Bubble Detection, Technical Indicators & ML Pipeline
**File:** `tests/unit/test_bubble_ml.py` &nbsp;|&nbsp; **Status:** ✅ PASSED &nbsp;|&nbsp; **37/37 tests passing**

<details>
<summary>Show all tests</summary>

| Test | Status |
|------|--------|
| `test_returns_all_keys` | ✅ PASSED |
| `test_bubble_score_in_unit_interval` | ✅ PASSED |
| `test_bubble_score_without_volume` | ✅ PASSED |
| `test_high_kurtosis_increases_score` | ✅ PASSED |
| `test_insufficient_returns_no_crash` | ✅ PASSED |
| `test_network_value_positive` | ✅ PASSED |
| `test_mmv_ratio_type` | ✅ PASSED |
| `test_bubble_regime_labels` | ✅ PASSED |
| `test_gph_returns_two_values` | ✅ PASSED |
| `test_gph_iid_d_near_zero` | ✅ PASSED |
| `test_gph_se_uses_asymptotic_formula` | ✅ PASSED |
| `test_returns_all_columns` | ✅ PASSED |
| `test_index_aligned` | ✅ PASSED |
| `test_sma20_after_warmup` | ✅ PASSED |
| `test_sma50_after_warmup` | ✅ PASSED |
| `test_rsi_bounded` | ✅ PASSED |
| `test_macd_histogram_equals_diff` | ✅ PASSED |
| `test_bollinger_upper_ge_lower` | ✅ PASSED |
| `test_rsi_wilder_ema` | ✅ PASSED |
| `test_returns_all_keys` | ✅ PASSED |
| `test_three_models_trained` | ✅ PASSED |
| `test_metrics_present` | ✅ PASSED |
| `test_rmse_non_negative` | ✅ PASSED |
| `test_last_features_scaled` | ✅ PASSED |
| `test_returns_none_for_short_series` | ✅ PASSED |
| `test_works_without_volume` | ✅ PASSED |
| `test_bullish_text` | ✅ PASSED |
| `test_bearish_text` | ✅ PASSED |
| `test_neutral_text` | ✅ PASSED |
| `test_mixed_text` | ✅ PASSED |
| `test_score_bounded` | ✅ PASSED |
| `test_returns_dataframe` | ✅ PASSED |
| `test_feature_columns` | ✅ PASSED |
| `test_one_row_per_asset` | ✅ PASSED |
| `test_sharpe_uses_rf` | ✅ PASSED |
| `test_max_drawdown_non_positive` | ✅ PASSED |
| `test_volatility_positive` | ✅ PASSED |

</details>

### Risk Score, Error Handling & Ticker Parser
**File:** `tests/unit/test_risk_and_errors.py` &nbsp;|&nbsp; **Status:** ✅ PASSED &nbsp;|&nbsp; **36/36 tests passing**

<details>
<summary>Show all tests</summary>

| Test | Status |
|------|--------|
| `test_score_in_range` | ✅ PASSED |
| `test_low_vix_low_score` | ✅ PASSED |
| `test_high_vix_raises_score` | ✅ PASSED |
| `test_inverted_yield_curve_raises_score` | ✅ PASSED |
| `test_gold_rally_raises_score` | ✅ PASSED |
| `test_empty_risk_data_returns_zero` | ✅ PASSED |
| `test_score_never_exceeds_100` | ✅ PASSED |
| `test_data_fetch_is_quantlab_error` | ✅ PASSED |
| `test_validation_is_quantlab_error` | ✅ PASSED |
| `test_calculation_is_quantlab_error` | ✅ PASSED |
| `test_export_is_quantlab_error` | ✅ PASSED |
| `test_fields_preserved` | ✅ PASSED |
| `test_user_message_defaults_to_message` | ✅ PASSED |
| `test_str_representation` | ✅ PASSED |
| `test_returns_value_on_success` | ✅ PASSED |
| `test_returns_none_on_data_fetch_error` | ✅ PASSED |
| `test_returns_none_on_calculation_error` | ✅ PASSED |
| `test_returns_none_on_generic_exception` | ✅ PASSED |
| `test_preserves_function_name` | ✅ PASSED |
| `test_passes_args_through` | ✅ PASSED |
| `test_space_separated` | ✅ PASSED |
| `test_comma_separated` | ✅ PASSED |
| `test_comma_space_mixed` | ✅ PASSED |
| `test_semicolon_separated` | ✅ PASSED |
| `test_forex_ticker` | ✅ PASSED |
| `test_crypto_ticker` | ✅ PASSED |
| `test_commodity_ticker` | ✅ PASSED |
| `test_index_ticker` | ✅ PASSED |
| `test_strips_special_chars` | ✅ PASSED |
| `test_empty_input` | ✅ PASSED |
| `test_uppercase_normalisation` | ✅ PASSED |
| `test_duplicate_tickers_preserved` | ✅ PASSED |
| `test_empty_tickers_raises` | ✅ PASSED |
| `test_network_error_raises_data_fetch_error` | ✅ PASSED |
| `test_rate_limit_raises_data_fetch_error` | ✅ PASSED |
| `test_empty_response_raises_data_fetch_error` | ✅ PASSED |

</details>

### End-to-End Integration Pipeline
**File:** `tests/unit/test_integration.py` &nbsp;|&nbsp; **Status:** ✅ PASSED &nbsp;|&nbsp; **16/16 tests passing**

<details>
<summary>Show all tests</summary>

| Test | Status |
|------|--------|
| `test_returns_computed_correctly` | ✅ PASSED |
| `test_metrics_all_tickers` | ✅ PASSED |
| `test_sharpe_formula` | ✅ PASSED |
| `test_full_optimization_roundtrip` | ✅ PASSED |
| `test_portfolio_metrics_from_optimized_weights` | ✅ PASSED |
| `test_bubble_aware_reduces_overvalued_weight` | ✅ PASSED |
| `test_pipeline_for_all_tickers` | ✅ PASSED |
| `test_indicators_for_all_tickers` | ✅ PASSED |
| `test_compute_asset_features_pipeline` | ✅ PASSED |
| `test_ml_all_models_predict` | ✅ PASSED |
| `test_last_features_can_predict` | ✅ PASSED |
| `test_predictions_actuals_length_match` | ✅ PASSED |
| `test_bs_to_greeks_to_payoff` | ✅ PASSED |
| `test_put_call_parity_pipeline` | ✅ PASSED |
| `test_bearish_sentiment_with_high_risk` | ✅ PASSED |
| `test_bullish_sentiment_low_risk_scenario` | ✅ PASSED |

</details>

### Streamlit UI — Widgets, Presets, Session State, Run Analysis
**File:** `tests/frontend/test_frontend.py` &nbsp;|&nbsp; **Status:** ✅ PASSED &nbsp;|&nbsp; **95/95 tests passing**

<details>
<summary>Show all tests</summary>

| Test | Status |
|------|--------|
| `test_app_loads_without_exception` | ✅ PASSED |
| `test_session_state_theme_defaults_to_light` | ✅ PASSED |
| `test_session_state_analysis_complete_false` | ✅ PASSED |
| `test_session_state_debug_mode_false` | ✅ PASSED |
| `test_session_state_last_updated_initialising` | ✅ PASSED |
| `test_session_state_data_is_none` | ✅ PASSED |
| `test_no_errors_on_fresh_load` | ✅ PASSED |
| `test_run_analysis_button_present` | ✅ PASSED |
| `test_run_analysis_button_not_yet_clicked` | ✅ PASSED |
| `test_dark_mode_toggle_exists` | ✅ PASSED |
| `test_dark_mode_defaults_to_off` | ✅ PASSED |
| `test_toggle_dark_mode_on_sets_theme` | ✅ PASSED |
| `test_toggle_dark_mode_off_resets_theme` | ✅ PASSED |
| `test_dark_mode_no_exception` | ✅ PASSED |
| `test_asset_class_selector_exists` | ✅ PASSED |
| `test_default_asset_class_is_stocks` | ✅ PASSED |
| `test_asset_class_options_present` | ✅ PASSED |
| `test_select_forex_class` | ✅ PASSED |
| `test_select_crypto_class` | ✅ PASSED |
| `test_select_commodities_class` | ✅ PASSED |
| `test_preset_selectbox_exists` | ✅ PASSED |
| `test_default_preset_is_none` | ✅ PASSED |
| `test_preset_options_include_tech_giants` | ✅ PASSED |
| `test_preset_options_include_crypto` | ✅ PASSED |
| `test_preset_options_include_forex` | ✅ PASSED |
| `test_selecting_tech_giants_updates_tickers` | ✅ PASSED |
| `test_selecting_crypto_preset_updates_tickers` | ✅ PASSED |
| `test_selecting_forex_preset_updates_tickers` | ✅ PASSED |
| `test_selecting_precious_metals_preset` | ✅ PASSED |
| `test_at_least_22_preset_options` | ✅ PASSED |
| `test_ticker_area_exists` | ✅ PASSED |
| `test_ticker_area_label` | ✅ PASSED |
| `test_default_tickers_non_empty` | ✅ PASSED |
| `test_set_tickers_space_separated` | ✅ PASSED |
| `test_set_tickers_comma_separated` | ✅ PASSED |
| `test_set_forex_tickers` | ✅ PASSED |
| `test_set_commodity_tickers` | ✅ PASSED |
| `test_empty_tickers_no_crash` | ✅ PASSED |
| `test_whitespace_only_no_crash` | ✅ PASSED |
| `test_mc_simulations_slider_exists` | ✅ PASSED |
| `test_mc_simulations_default_500` | ✅ PASSED |
| `test_mc_simulations_min_100_max_2000` | ✅ PASSED |
| `test_set_mc_simulations_1000` | ✅ PASSED |
| `test_forecast_days_slider_exists` | ✅ PASSED |
| `test_forecast_days_default_90` | ✅ PASSED |
| `test_simulation_method_selectbox` | ✅ PASSED |
| `test_switch_to_gbm_simulation` | ✅ PASSED |
| `test_bubble_aware_checkbox_exists` | ✅ PASSED |
| `test_bubble_aware_defaults_true` | ✅ PASSED |
| `test_disable_bubble_aware` | ✅ PASSED |
| `test_bubble_penalty_slider_exists` | ✅ PASSED |
| `test_bubble_penalty_default_0_5` | ✅ PASSED |
| `test_benchmark_selectbox_exists` | ✅ PASSED |
| `test_benchmark_options` | ✅ PASSED |
| `test_rebalancing_selectbox_exists` | ✅ PASSED |
| `test_rebalancing_default_buy_hold` | ✅ PASSED |
| `test_rebalancing_options` | ✅ PASSED |
| `test_custom_rf_checkbox` | ✅ PASSED |
| `test_enable_custom_rf` | ✅ PASSED |
| `test_ml_training_period_slider_exists` | ✅ PASSED |
| `test_ml_training_period_default_3` | ✅ PASSED |
| `test_ml_training_period_range` | ✅ PASSED |
| `test_clustering_method_selectbox` | ✅ PASSED |
| `test_switch_clustering_to_gmm` | ✅ PASSED |
| `test_anomaly_sensitivity_slider` | ✅ PASSED |
| `test_chart_height_slider_exists` | ✅ PASSED |
| `test_chart_height_default_500` | ✅ PASSED |
| `test_chart_height_range` | ✅ PASSED |
| `test_export_sections_multiselect_exists` | ✅ PASSED |
| `test_export_sections_has_defaults` | ✅ PASSED |
| `test_export_sections_all_options_available` | ✅ PASSED |
| `test_auto_refresh_toggle_exists` | ✅ PASSED |
| `test_auto_refresh_defaults_off` | ✅ PASSED |
| `test_enable_auto_refresh` | ✅ PASSED |
| `test_debug_checkbox_exists` | ✅ PASSED |
| `test_debug_mode_defaults_off` | ✅ PASSED |
| `test_enable_debug_mode` | ✅ PASSED |
| `test_disable_debug_mode_via_new_session` | ✅ PASSED |
| `test_analysis_not_complete_before_click` | ✅ PASSED |
| `test_analysis_button_present_and_clickable` | ✅ PASSED |
| `test_data_is_none_before_analysis` | ✅ PASSED |
| `test_run_empty_tickers_shows_error_and_stays_incomplete` | ✅ PASSED |
| `test_simulation_method_default_behavioral_agent` | ✅ PASSED |
| `test_simulation_method_switch_to_gbm` | ✅ PASSED |
| `test_clustering_method_persists_across_runs` | ✅ PASSED |
| `test_mc_sims_persists_after_preset_change` | ✅ PASSED |
| `test_empty_tickers_shows_error` | ✅ PASSED |
| `test_whitespace_only_tickers_shows_error` | ✅ PASSED |
| `test_error_does_not_persist_after_ticker_fix` | ✅ PASSED |
| `test_analysis_complete_false_on_fresh_load` | ✅ PASSED |
| `test_no_error_on_fresh_load` | ✅ PASSED |
| `test_preset_then_dark_mode` | ✅ PASSED |
| `test_change_multiple_advanced_settings` | ✅ PASSED |
| `test_forex_preset_with_benchmark_change` | ✅ PASSED |
| `test_debug_mode_with_dark_mode` | ✅ PASSED |

</details>

## 🚀 Running the Tests

### Quick start
```bash
pip install -r requirements.txt && pip install pytest pytest-cov
python run_tests.py
```

### Options
```bash
python run_tests.py --fast           # skip integration tests
python run_tests.py --module options  # run only test_options.py
python run_tests.py --no-cov         # skip coverage (faster)
python run_tests.py --out my_qa.md   # custom output path
```

### Makefile shortcuts
```bash
make qa            # full suite + QA-REPORT.md
make test          # pytest verbose
make fast          # skip integration
make coverage      # HTML coverage + auto-open
make t-portfolio   # run only portfolio tests
make lint          # flake8
make clean         # remove test artifacts
```

## 🏗️ Test Architecture

```
tests/
├── conftest.py                    # Top-level markers only
│
├── unit/                          # Backend tests — use Streamlit stub
│   ├── conftest.py                # Streamlit stub + synthetic OHLCV fixtures
│   ├── test_valuation.py          # CAPM, Beta (ddof=1), WACC, DCF, FF, APT
│   ├── test_portfolio.py          # 9 strategies × 3 invariants (parametrized)
│   │                              # Risk Parity, HRP, bubble-aware penalty
│   ├── test_options.py            # B-S known value, put-call parity, all 5 Greeks
│   ├── test_bubble_ml.py          # BubbleDetector, GPH SE, RSI Wilder's EMA,
│   │                              # MACD histogram, ML pipeline, sentiment
│   ├── test_risk_and_errors.py    # Risk score, exception hierarchy, ticker parser
│   └── test_integration.py        # End-to-end: prices → portfolio → ML → options
│
└── frontend/                      # Streamlit UI tests — use real AppTest
    ├── conftest.py                # Synthetic price/volume fixtures
    └── test_frontend.py           # 95 tests: widgets, presets, session state,
                                   # dark mode, advanced settings, error handling
```

Unit tests run **fully offline** — no Yahoo Finance calls, no Streamlit server.
Frontend tests use `streamlit.testing.v1.AppTest` — real Streamlit runtime, no browser.

---
*Generated by `run_tests.py` · 2026-04-02 19:41 UTC*
