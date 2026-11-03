import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import enki
from enki import backtest, forecast
from dataclasses import replace

from enki.backtest import BacktestSpec
from enki.config import ModelConfig, RunOptions, beta_prior_table
from enki.data import NSW_STATE_RESULTS, prepare_booths, state_matrix
from enki.onp import apply_onp
from enki.pipeline import check_trace
from enki.polls import _pct_scale, draw_statewide, summarise_polls
from enki.share_model import check_layers, simulate_shares
from enki.volume_model import projected_votes_from_electors, resolve_volumes
from fake_fit import fake_fit_factory, fake_volume_fit, make_backtest_polls, make_booths, make_master_polls, make_polls


def patched(D=200):
    f = fake_fit_factory(D)
    return mock.patch.multiple("enki.forecast", fit_share_model=f), mock.patch.multiple("enki.backtest", fit_share_model=f)


class Base(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig(prediction_date="2026-11-01")
        self.booths = make_booths()
        self.df = prepare_booths(self.booths, self.cfg)
        self.fit = fake_fit_factory(300)(self.df, self.cfg, NSW_STATE_RESULTS)
        self.rng = np.random.default_rng(1)


class TestConfig(Base):
    def test_unknown_key_suggestion_and_renames(self):
        with self.assertRaises(KeyError) as e:
            enki.resolve({"booth_results": self.booths, "polls": make_polls(), "draw": 10})
        self.assertIn("did you mean 'draws'", str(e.exception))
        with self.assertRaises(KeyError) as e:
            enki.resolve({"booth_results": self.booths, "polls": make_polls(), "brian_mode": True})
        self.assertIn("onp_mode='brian'", str(e.exception))

    def test_auto_scales_are_none_not_sentinels(self):
        self.assertIsNone(ModelConfig().sigma_d_median)
        self.assertIsNone(ModelConfig().sigma_eps_sd)

    def test_validation_collects_errors(self):
        with self.assertRaises(ValueError) as e:
            enki.resolve({"booth_results": self.booths, "polls": make_polls(), "onp_mode": "nope", "tide_process": "x", "beta_priors": {"OTHER": (0, 1)}})
        for frag in ("onp_mode", "tide_process", "beta_priors"):
            self.assertIn(frag, str(e.exception))

    def test_beta_priors_single_source(self):
        t = beta_prior_table(ModelConfig(beta_priors={"ALP": (0.5, 0.1)}))
        self.assertEqual(t["ALP"], (0.5, 0.1))
        self.assertEqual(t["GRN"], (0.3, 0.35))
        self.assertNotIn("OTHER", t)


class TestData(Base):
    def test_prepare(self):
        self.assertTrue((self.df["n"] > 0).all())
        self.assertTrue((self.df[[f"y_{p}" for p in self.cfg.parties]].sum(axis=1) == self.df["n"]).all())
        self.assertEqual(self.df.loc[self.df["booth"] == "Pre-poll", "btype"].unique().tolist(), [1])

    def test_state_matrix_sums_to_100(self):
        S = state_matrix(NSW_STATE_RESULTS, self.cfg.parties, [2019, 2023])
        np.testing.assert_allclose(S.sum(axis=1), 100.0)


class TestPolls(Base):
    def test_scale_decided_per_table_not_per_column(self):
        p = make_polls(onp_fraction=True)                     # ONP never exceeds 1.0 point; others are in percent
        self.assertEqual(_pct_scale(p, list(self.cfg.parties) + ["ONP"]), 1.0)
        cfg = ModelConfig(prediction_date="2026-11-01", onp_mode="entry")
        s = summarise_polls(p, cfg, NSW_STATE_RESULTS)
        self.assertLess(s.onp_central, 2.0)                   # a per-column heuristic would have made this ~80

    def test_fractions_table_is_scaled(self):
        p = make_polls(onp=False)
        p[list(self.cfg.parties)] = p[list(self.cfg.parties)] / 100.0
        s = summarise_polls(p, self.cfg, NSW_STATE_RESULTS)
        self.assertAlmostEqual(s.central.sum(), 100.0)
        self.assertGreater(s.central[0], 30.0)

    def test_entry_and_denominator_require_onp_polls(self):
        for kw in ({"onp_mode": "entry"}, {"onp_state_denominator": True}):
            cfg = ModelConfig(prediction_date="2026-11-01", **kw)
            with self.assertRaises(ValueError):
                summarise_polls(make_polls(onp=False), cfg, NSW_STATE_RESULTS)
            with self.assertRaises(ValueError):
                summarise_polls(enki.PollInput(38.0, 36.0, 40.0), cfg, NSW_STATE_RESULTS)

    def test_draws_match_poll_average(self):
        s = summarise_polls(make_polls(onp=False), self.cfg, NSW_STATE_RESULTS)
        d = draw_statewide(s, self.cfg, 20000, self.rng)
        np.testing.assert_allclose(d.S.sum(axis=1), 100.0)
        np.testing.assert_allclose(d.S.mean(axis=0), s.central, atol=0.3)
        self.assertAlmostEqual(d.S[:, 0].std(), d.se_parts[3], delta=0.15)   # Dirichlet sd on the target matches the error budget

    def test_poll_input_path(self):
        s = summarise_polls(enki.PollInput(38.0, 36.0, 40.0), self.cfg, NSW_STATE_RESULTS)
        self.assertAlmostEqual(s.central.sum(), 100.0)
        self.assertAlmostEqual(s.central[0], 38.0)


class TestShares(Base):
    def _S(self, D):
        return np.tile([37.0, 34.0, 10.0, 19.0], (D, 1))

    def test_rows_sum_to_one(self):
        sim = simulate_shares(self.fit, self._S(self.fit.n_draws), self.cfg, self.rng)
        np.testing.assert_allclose(sim.p.sum(axis=2), 1.0)
        self.assertEqual(sim.gamma.shape, (self.fit.n_draws, 3))

    def test_unknown_layer_is_an_error(self):
        with self.assertRaises(ValueError):
            check_layers(("param",))
        with self.assertRaises(ValueError):
            simulate_shares(self.fit, self._S(self.fit.n_draws), self.cfg, self.rng, freeze=("tides",))

    def test_freezing_layers_narrows_and_keeps_random_stream(self):
        S = self._S(self.fit.n_draws)
        full = simulate_shares(self.fit, S, self.cfg, np.random.default_rng(5)).p[:, :, 0]
        nob = simulate_shares(self.fit, S, self.cfg, np.random.default_rng(5), freeze=("booth",)).p[:, :, 0]
        allf = simulate_shares(self.fit, S, self.cfg, np.random.default_rng(5), freeze=("params", "tide", "booth")).p[:, :, 0]
        self.assertLess(nob.std(), full.std())
        self.assertLess(allf.std(axis=0).max(), 1e-12)   # no spread across draws (booths still differ by their mean offsets)

    def test_correlated_shocks_are_used_in_the_forecast(self):
        D = 20000
        fit = fake_fit_factory(D)(self.df, ModelConfig(prediction_date="2026-11-01"), NSW_STATE_RESULTS)
        K = len(fit.parties)
        fit.draws["d"][:] = 0.0
        fit.draws["rho"][:] = 0.0
        fit.draws["sigma_d"][:] = 1.0
        L = np.linalg.cholesky(np.array([[1, .9, 0, 0], [.9, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]))
        fit.draws["corr_chol"] = np.tile(L, (D, 1, 1))
        sim = simulate_shares(fit, np.tile([37.0, 34.0, 10.0, 19.0], (D, 1)), ModelConfig(), self.rng)
        self.assertAlmostEqual(np.corrcoef(sim.d_next[:, 0], sim.d_next[:, 1])[0, 1], 0.9, delta=0.02)
        self.assertAlmostEqual(np.corrcoef(sim.d_next[:, 0], sim.d_next[:, 2])[0, 1], 0.0, delta=0.03)

    def test_random_walk_adds_variance(self):
        S = self._S(self.fit.n_draws)
        ar = simulate_shares(self.fit, S, self.cfg, np.random.default_rng(2)).d_next
        rw = simulate_shares(self.fit, S, ModelConfig(tide_process="random_walk"), np.random.default_rng(2)).d_next
        self.assertGreater(rw.std(), 0.0)
        self.assertNotAlmostEqual(float(ar.std()), float(rw.std()), places=3)


class TestOnp(Base):
    def setUp(self):
        super().setUp()
        D = self.fit.n_draws
        self.p = simulate_shares(self.fit, np.tile([37.0, 34.0, 10.0, 19.0], (D, 1)), self.cfg, self.rng).p

    def test_none(self):
        r = apply_onp(self.p, None, self.cfg.parties, 0, self.cfg, self.rng)
        self.assertIs(r.p, self.p)
        np.testing.assert_array_equal(r.target_scale, 1.0)

    def test_brian_conserves_mass_and_scale(self):
        cfg = ModelConfig(onp_mode="brian", brian_transfer={"LNP": 0.2, "ALP": 0.05})
        r = apply_onp(self.p, None, cfg.parties, 0, cfg, self.rng)
        np.testing.assert_allclose(r.p.sum(axis=2), 1.0)
        self.assertEqual(r.out_parties[-1], "ONP")
        np.testing.assert_allclose(r.p[:, :, 0], self.p[:, :, 0] * r.target_scale[:, None])
        self.assertAlmostEqual(r.target_scale[0], 0.95)

    def test_entry_conserves_mass_and_scale(self):
        cfg = ModelConfig(onp_mode="entry")
        onp = np.full(self.p.shape[0], 8.0)
        r = apply_onp(self.p, onp, cfg.parties, 0, cfg, self.rng)
        np.testing.assert_allclose(r.p.sum(axis=2), 1.0)
        np.testing.assert_allclose(r.p[:, :, 0], self.p[:, :, 0] * r.target_scale[:, None])
        np.testing.assert_allclose(r.p[:, 0, -1], r.seat_share)


class TestVolumes(Base):
    def test_electors_path_totals(self):
        cfg = ModelConfig(registered_electors=60000, expected_turnout_rate=0.9)
        w = projected_votes_from_electors(self.df, cfg)
        informal = self.df["informal_votes"].sum() / self.df["total_votes"].sum()
        self.assertAlmostEqual(w.sum(), 60000 * 0.9 * (1 - informal), places=4)
        r = resolve_volumes(self.df, self.fit.booth_names, cfg, 50, self.rng)
        self.assertEqual(r.mode, "electors")
        self.assertTrue(r.active.all())
        np.testing.assert_allclose(r.V.sum(axis=1), w.sum())

    def test_projected_wins_and_inactive_booth_dropped(self):
        names = self.fit.booth_names
        proj = pd.Series({b: 1000.0 for b in names})
        proj[names[0]] = 0.0
        r = resolve_volumes(self.df, names, ModelConfig(registered_electors=1), 10, self.rng, projected=proj)
        self.assertEqual(r.mode, "projected")
        self.assertFalse(r.active[0])


class TestForecast(Base):
    def _run(self, **kw):
        cfg = ModelConfig(prediction_date="2026-11-01", registered_electors=60000, **kw)
        a, _ = patched(300)
        with a:
            return forecast.run_forecast(prepare_booths(self.booths, cfg), NSW_STATE_RESULTS, make_polls(), cfg), cfg

    def test_seat_draws_sum_to_100_and_trace_complete(self):
        res, _ = self._run()
        np.testing.assert_allclose(res.draws_df.sum(axis=1), 100.0)
        self.assertEqual(check_trace(res.trace), [])
        self.assertEqual(len(res.seat), 3)
        self.assertEqual(len(res.attribution), 7)

    def test_all_layers_off_is_a_point(self):
        res, _ = self._run()
        self.assertLess(res.attribution.iloc[-1]["width_90"], 1e-9)
        self.assertGreater(res.attribution.iloc[0]["width_90"], 0.0)

    def test_onp_modes_add_a_party_and_conserve(self):
        for kw in ({"onp_mode": "brian"}, {"onp_mode": "entry"}):
            res, _ = self._run(**kw)
            self.assertEqual(res.party_names[-1], "ONP")
            np.testing.assert_allclose(res.draws_df.sum(axis=1), 100.0)
            np.testing.assert_allclose(res.booth_party_draws.sum(axis=2), 100.0)
        # attribution of the target uses the same scale: all-on seat median equals the seat table's median
        res, _ = self._run(onp_mode="brian")
        self.assertAlmostEqual(res.attribution.iloc[0]["p50"], res.draws_df["ALP"].median(), delta=0.35)

    def test_booth_forecast_matches_party_draws(self):
        res, _ = self._run(onp_mode="brian")
        np.testing.assert_allclose(res.booth_draws_df.values, res.booth_party_draws[:, :, 0])

    def test_no_tide_row_for_reference_target(self):
        a, _ = patched(100)
        cfg = ModelConfig(prediction_date="2026-11-01", registered_electors=60000, party="OTHER")
        with a:
            res = forecast.run_forecast(prepare_booths(self.booths, cfg), NSW_STATE_RESULTS, make_polls(), cfg)
        self.assertTrue(res.seat["gamma_next"].isna().all())


class TestBacktest(Base):
    def test_crps_matches_bruteforce(self):
        r = np.random.default_rng(3)
        d, a = r.normal(size=(40, 5)), r.normal(size=5)
        brute = np.array([np.abs(d[:, j] - a[j]).mean() - 0.5 * np.abs(d[:, j][:, None] - d[:, j][None, :]).mean() for j in range(5)])
        np.testing.assert_allclose(backtest.crps_ensemble(d, a), brute, atol=1e-12)
        self.assertAlmostEqual(float(backtest.crps_ensemble(np.full((10, 1), 2.0), np.array([5.0]))[0]), 3.0)   # point forecast -> MAE

    def test_coverage_known_answer(self):
        draws = np.tile(np.arange(101.0)[:, None], (1, 4))               # 0..100 for every booth
        cov = backtest.interval_coverage(draws, np.array([50.0, 50.0, 2.0, 99.0]), (50, 90))
        self.assertEqual(cov["cover_50"], 0.5)
        self.assertEqual(cov["cover_90"], 0.5)
        self.assertAlmostEqual(cov["bias_50"], 0.0)

    def test_walk_forward_table(self):
        _, b = patched(300)
        with b:
            r = backtest.backtest_shares(self.df, NSW_STATE_RESULTS, self.cfg, return_draws=True)
        t = r.table
        self.assertEqual(t["target"].tolist(), [2015, 2019, 2023])
        for c in ("booth_MAE", "booth_CRPS", "booth_cover_50", "booth_cover_80", "booth_cover_90", "booth_cover_95",
                  "nochange_MAE", "statewide_swing_MAE", "model_MAE_common", "model_seat_err_common", "nochange_seat_err"):
            self.assertIn(c, t.columns)
        self.assertTrue(t[[f"booth_cover_{k}" for k in (50, 80, 90, 95)]].apply(lambda s: s.between(0, 1)).all().all())
        self.assertTrue((t["booth_cover_50"] <= t["booth_cover_95"]).all())
        d = r.draws[2023]
        self.assertEqual(d["booth_party_draws"].shape[2], 4)
        np.testing.assert_allclose(d["booth_party_draws"].sum(axis=2), 100.0)
        self.assertEqual(d["actual_party_votes"].sum(axis=1).tolist(), d["actual_n"].astype(int).tolist())

    def test_single_target_reproduces_its_row(self):
        _, b = patched(300)
        with b:
            full = backtest.backtest_shares(self.df, NSW_STATE_RESULTS, self.cfg).table
            one = backtest.backtest_shares(self.df, NSW_STATE_RESULTS, self.cfg, targets=[2023]).table
        pd.testing.assert_series_equal(full[full["target"] == 2023].iloc[0], one.iloc[0], check_names=False)


class TestBacktestModes(Base):
    def _run(self, cfg=None, **spec_kw):
        cfg = cfg or ModelConfig(prediction_date="2026-11-01", poll_house_sd=6.0)
        df = prepare_booths(self.booths, cfg)
        _, b = patched(300)
        with b, mock.patch("enki.backtest.fit_volume_model", fake_volume_fit):
            return backtest.backtest_shares(df, NSW_STATE_RESULTS, cfg, spec=BacktestSpec(**spec_kw)).table

    def test_modes_recorded_and_oracle_has_no_poll_columns(self):
        t = self._run()
        self.assertEqual(set(t["state_mode"]), {"oracle"})
        self.assertNotIn("state_error", t.columns)

    def test_polls_mode_measures_poll_error_and_widens_the_band(self):
        cfg = ModelConfig(prediction_date="2026-11-01", poll_house_sd=6.0)
        bp = make_backtest_polls(NSW_STATE_RESULTS, cfg)
        oracle, polls = self._run(cfg), self._run(cfg, state_mode="polls", polls=bp)
        self.assertEqual(set(polls["state_mode"]), {"polls"})
        self.assertTrue(polls["state_error"].between(0.5, 2.0).all())                 # ALP was over-polled by 1.5 points before renormalising
        self.assertTrue(np.isfinite(polls["state_z"]).all())
        self.assertGreater(polls["seat_width"].mean(), oracle["seat_width"].mean())   # poll error now reaches the seat

    def test_polls_horizon_changes_the_forecast_lead_time(self):
        cfg = ModelConfig(prediction_date="2026-11-01")
        bp = make_backtest_polls(NSW_STATE_RESULTS, cfg)
        latest = self._run(cfg, state_mode="polls", polls=bp)                              # forecast made at the last poll (30 days out)
        early = self._run(cfg, state_mode="polls", polls=bp, horizon_days=100.0)           # made 100 days out, so only the 150-day-out poll counts
        self.assertTrue((early["state_se_used"] > latest["state_se_used"]).all())          # longer lead time and fewer polls: more error

    def test_noised_mode_uses_the_error_budget(self):
        cfg = ModelConfig(prediction_date="2026-11-01", poll_house_sd=2.0, time_horizon_gamma=1.0)
        t = self._run(cfg, state_mode="noised", horizon_days=365.0)
        np.testing.assert_allclose(t["state_se_used"], np.sqrt(2.0 ** 2 + 1.0 ** 2), rtol=1e-6)

    def test_bad_polls_input_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            BacktestSpec(state_mode="polls", polls=None)
        with self.assertRaises(ValueError):
            BacktestSpec(state_mode="sometimes")
        cfg = ModelConfig(prediction_date="2026-11-01")
        bp = make_backtest_polls(NSW_STATE_RESULTS, cfg, years=(2015, 2019))              # 2023 missing
        with self.assertRaises(ValueError) as e:
            self._run(cfg, state_mode="polls", polls=bp)
        self.assertIn("2023", str(e.exception))

    def test_modelled_volumes_run_and_keep_the_actual_seat(self):
        cfg = ModelConfig(prediction_date="2026-11-01")
        a, m = self._run(cfg), self._run(cfg, volume_mode="modelled")
        self.assertEqual(set(m["volume_mode"]), {"modelled"})
        np.testing.assert_allclose(a["seat_actual"], m["seat_actual"])
        self.assertTrue(np.isfinite(m["seat_width"]).all())

    def test_ENKI_flags_the_optimistic_default_and_requires_polls(self):
        a, b = patched(100)
        base = {"booth_results": self.booths, "polls": make_polls(), "registered_electors": 60000, "prediction_date": "2026-11-01",
                "run_forecast": False, "print_report": False, "verbose": False, "backtest_targets": [2023]}
        with a, b:
            y = enki.ENKI(base)
            self.assertTrue(any("TRUE statewide result" in w for w in y.warnings))
            with self.assertRaises(ValueError):                       # polls mode is strict: the 2026-only table has no polls for 2023
                enki.ENKI({**base, "backtest_state_mode": "polls"})
            with self.assertRaises(KeyError):                         # and a single poll NUMBER cannot be backtested at all
                enki.ENKI({**base, "polls": {"central": 38, "low": 36, "high": 40}, "backtest_state_mode": "polls"})
            y2 = enki.ENKI({**base, "backtest_state_mode": "polls", "backtest_polls": make_backtest_polls(NSW_STATE_RESULTS, ModelConfig())})
            self.assertFalse(any("TRUE statewide result" in w for w in y2.warnings))
            self.assertIn("state_error", y2.backtest.columns)
        with self.assertRaises(ValueError):
            enki.resolve({"booth_results": self.booths, "polls": make_polls(), "backtest_state_mode": "nope"})


class TestAutoPolls(Base):
    def _run(self, polls, cfg=None, **spec_kw):
        cfg = cfg or ModelConfig(prediction_date="2026-11-01")
        df = prepare_booths(self.booths, cfg)
        _, b = patched(300)
        with b:
            return backtest.backtest_shares(df, NSW_STATE_RESULTS, cfg, spec=BacktestSpec(polls=polls, **spec_kw)).table.set_index("target")

    def test_default_is_auto(self):
        self.assertEqual(RunOptions().backtest_state_mode, "auto")
        self.assertEqual(BacktestSpec().state_mode, "auto")

    def test_uses_polls_where_they_exist_and_the_true_result_where_not(self):
        cfg = ModelConfig(prediction_date="2026-11-01")
        t = self._run(make_master_polls(cfg, NSW_STATE_RESULTS), cfg)                    # polls for the 2019 and 2023 cycles only
        self.assertEqual(t["state_mode"].to_dict(), {2015: "oracle", 2019: "polls", 2023: "polls"})
        self.assertTrue(t.loc[[2019, 2023], "state_error"].between(0.3, 2.5).all())
        self.assertTrue(np.isnan(t.loc[2015, "state_error"]))
        self.assertEqual(t.loc[2015, "n_polls"], 0)

    def test_no_leakage_between_cycles(self):
        cfg = ModelConfig(prediction_date="2026-11-01")
        t = self._run(make_master_polls(cfg, NSW_STATE_RESULTS), cfg)
        self.assertEqual(t.loc[2019, "n_polls"], 3)          # not the 2023-cycle polls, not the post-2023 one
        self.assertEqual(t.loc[2023, "n_polls"], 3)          # the 2023-05-01 poll (after the election) and the 2026 polls are excluded
        self.assertNotEqual(t.loc[2023, "poll_central"], t.loc[2019, "poll_central"])
        # a poll the day AFTER an election never counts for it
        import pandas as pd
        from enki.backtest import cycle_polls
        pl = make_master_polls(cfg, NSW_STATE_RESULTS)
        spec = BacktestSpec(polls=pl)
        sub, elec = cycle_polls(spec, 2023)
        self.assertTrue((pd.to_datetime(sub["date"]) <= elec).all())
        self.assertTrue((pd.to_datetime(sub["date"]) > pd.Timestamp("2019-03-23")).all())

    def test_a_file_with_only_post_2023_polls_never_applies(self):
        cfg = ModelConfig(prediction_date="2026-11-01")
        t = self._run(make_polls(onp=False), cfg)             # polls dated 2026 only
        self.assertEqual(set(t["state_mode"]), {"oracle"})
        self.assertEqual(set(t["n_polls"]), {0})

    def test_min_polls_threshold_and_missing_requirement_fall_back(self):
        cfg = ModelConfig(prediction_date="2026-11-01")
        pl = make_master_polls(cfg, NSW_STATE_RESULTS)
        self.assertEqual(set(self._run(pl, cfg, min_polls=4)["state_mode"]), {"oracle"})
        # ONP required but never polled in those cycles -> auto falls back instead of crashing
        cfg2 = ModelConfig(prediction_date="2026-11-01", onp_mode="entry")
        t = self._run(pl, cfg2)
        self.assertEqual(set(t["state_mode"]), {"oracle"})
        self.assertIn("ONP", " ".join(t["state_note"].dropna()))

    def test_polls_mode_is_strict_and_horizon_applies(self):
        cfg = ModelConfig(prediction_date="2026-11-01")
        pl = make_master_polls(cfg, NSW_STATE_RESULTS)
        with self.assertRaises(ValueError) as e:
            self._run(pl, cfg, state_mode="polls")            # no polls for 2015
        self.assertIn("2015", str(e.exception))
        late = self._run(pl, cfg, horizon_days=45.0)          # keeps only polls >= 45 days out (the 20-day-out ones drop)
        self.assertEqual(late.loc[2019, "n_polls"], 2)

    def test_election_dates_override_and_missing_date_is_an_error(self):
        cfg = ModelConfig(prediction_date="2026-11-01")
        pl = make_master_polls(cfg, NSW_STATE_RESULTS)
        with self.assertRaises(ValueError) as e:
            self._run(pl, cfg, election_dates={2019: "2019-03-23"})      # the 2015 and 2023 dates are unknown
        self.assertIn("election_dates", str(e.exception))

    def test_ENKI_uses_one_master_file_for_both_and_lists_the_fallbacks(self):
        a, b = patched(100)
        cfg = ModelConfig()
        master = make_master_polls(cfg, NSW_STATE_RESULTS)
        with a, b:
            y = enki.ENKI({"booth_results": self.booths, "polls": master, "registered_electors": 60000, "prediction_date": "2026-11-01",
                           "election_date": "2027-03-27", "print_report": False, "verbose": False})
        self.assertEqual(y.backtest.set_index("target")["state_mode"].to_dict(), {2015: "oracle", 2019: "polls", 2023: "polls"})
        self.assertTrue(any("[2015]" in w and "TRUE statewide result" in w for w in y.warnings))
        self.assertEqual(y.data_sources["backtest_polls"], "<same table as polls>")
        self.assertIsNotNone(y.forecast)

class TestEndToEnd(Base):
    def test_ENKI_writes_everything(self):
        a, b = patched(150)
        with tempfile.TemporaryDirectory() as d, a, b:
            y = enki.ENKI({"booth_results": self.booths, "polls": make_polls(), "seat_name": "Test Seat", "registered_electors": 60000,
                           "prediction_date": "2026-11-01", "output_root": d, "onp_mode": "brian", "plot_dpi": 40, "print_report": False,
                           "verbose": False, "backtest_targets": [2019, 2023]})
            out = Path(y.output_dir)
            for f in ("seat_primary_summary.csv", "forecast.csv", "params.csv", "backtest.csv", "backtest_draws_2023.npz", "booth_party_draws.npz",
                      "actual_primaries.csv", "inputs_resolved.json", "trace.npz", "run_meta.json", "backtest_summary_table.png",
                      "seat_distribution.png", "ridgeline.png", "booth_swing_scatter.png", "booth_volume_ridgeline.png", "uncertainty_attribution.csv"):
                self.assertTrue((out / f).exists(), f)
            self.assertEqual([w for w in y.warnings if "failed" in w], [])
            self.assertTrue((out.parent / "history.csv").exists())
            self.assertEqual(json.loads((out / "inputs_resolved.json").read_text())["config"]["onp_mode"], "brian")
            self.assertEqual(list(y.seat_summary.index), ["ALP", "LNP", "GRN", "OTHER", "ONP"])

    def test_forecast_only_no_backtest_no_output(self):
        a, _ = patched(100)
        with a:
            y = enki.ENKI({"booth_results": self.booths, "polls": {"central": 38, "low": 36, "high": 40}, "registered_electors": 60000,
                           "prediction_date": "2026-11-01", "run_backtest": False, "print_report": False, "verbose": False})
        self.assertIsNone(y.backtest)
        self.assertAlmostEqual(y.seat_summary["mean"].sum(), 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
