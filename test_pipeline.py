"""
Integration tests for features.build_model_frame and the analyze_fraud
pipeline functions (score_transactions, summarize_results).

All tests use small in-memory DataFrames so no file I/O is required.
"""
import pandas as pd
import pytest
from features import build_model_frame
from analyze_fraud import score_transactions, summarize_results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_transactions():
    return pd.DataFrame([
        # clean domestic low-risk transaction
        dict(transaction_id=1, account_id=10, amount_usd=50.0,
             device_risk_score=5, ip_country="US", is_international=0,
             velocity_24h=1, failed_logins_24h=0, timestamp="2026-01-01"),
        # high-risk: international, high device score, high velocity, failed logins
        dict(transaction_id=2, account_id=20, amount_usd=1500.0,
             device_risk_score=80, ip_country="NG", is_international=1,
             velocity_24h=8, failed_logins_24h=6, timestamp="2026-01-01"),
        # medium-risk: large domestic amount, mid device score
        dict(transaction_id=3, account_id=10, amount_usd=800.0,
             device_risk_score=50, ip_country="US", is_international=0,
             velocity_24h=2, failed_logins_24h=0, timestamp="2026-01-01"),
    ])


@pytest.fixture
def sample_accounts():
    return pd.DataFrame([
        dict(account_id=10, customer_name="Alice", country="US",
             signup_date="2022-01-01", kyc_level="full",
             account_age_days=730, prior_chargebacks=0, is_vip="Y"),
        dict(account_id=20, customer_name="Bob", country="NG",
             signup_date="2025-12-01", kyc_level="basic",
             account_age_days=14, prior_chargebacks=3, is_vip="N"),
    ])


@pytest.fixture
def sample_chargebacks():
    return pd.DataFrame([
        dict(transaction_id=2, chargeback_date="2026-02-01",
             chargeback_reason="card_not_present", loss_amount_usd=1500.0),
    ])


# ---------------------------------------------------------------------------
# build_model_frame
# ---------------------------------------------------------------------------

class TestBuildModelFrame:
    def test_joins_accounts_on_account_id(self, sample_transactions, sample_accounts):
        df = build_model_frame(sample_transactions, sample_accounts)
        assert "prior_chargebacks" in df.columns
        assert "kyc_level" in df.columns

    def test_no_rows_dropped_for_known_accounts(self, sample_transactions, sample_accounts):
        df = build_model_frame(sample_transactions, sample_accounts)
        assert len(df) == len(sample_transactions)

    def test_prior_chargebacks_joined_correctly(self, sample_transactions, sample_accounts):
        df = build_model_frame(sample_transactions, sample_accounts)
        row = df[df["account_id"] == 20].iloc[0]
        assert row["prior_chargebacks"] == 3

    def test_is_large_amount_true_at_threshold(self, sample_transactions, sample_accounts):
        df = build_model_frame(sample_transactions, sample_accounts)
        large = df[df["amount_usd"] >= 1000]
        assert (large["is_large_amount"] == 1).all()

    def test_is_large_amount_false_below_threshold(self, sample_transactions, sample_accounts):
        df = build_model_frame(sample_transactions, sample_accounts)
        small = df[df["amount_usd"] < 1000]
        assert (small["is_large_amount"] == 0).all()

    def test_login_pressure_categories(self, sample_transactions, sample_accounts):
        df = build_model_frame(sample_transactions, sample_accounts)
        # 0 logins → "none", 6 logins → "high"
        row_no_logins = df[df["failed_logins_24h"] == 0].iloc[0]
        row_many_logins = df[df["failed_logins_24h"] == 6].iloc[0]
        assert str(row_no_logins["login_pressure"]) == "none"
        assert str(row_many_logins["login_pressure"]) == "high"

    def test_unknown_account_id_produces_null_join(self, sample_accounts):
        tx_unknown = pd.DataFrame([
            dict(transaction_id=99, account_id=999, amount_usd=100.0,
                 device_risk_score=10, ip_country="US", is_international=0,
                 velocity_24h=1, failed_logins_24h=0, timestamp="2026-01-01"),
        ])
        df = build_model_frame(tx_unknown, sample_accounts)
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["prior_chargebacks"])


# ---------------------------------------------------------------------------
# score_transactions
# ---------------------------------------------------------------------------

class TestScoreTransactions:
    def test_produces_risk_score_column(self, sample_transactions, sample_accounts):
        scored = score_transactions(sample_transactions, sample_accounts)
        assert "risk_score" in scored.columns

    def test_produces_risk_label_column(self, sample_transactions, sample_accounts):
        scored = score_transactions(sample_transactions, sample_accounts)
        assert "risk_label" in scored.columns

    def test_all_rows_scored(self, sample_transactions, sample_accounts):
        scored = score_transactions(sample_transactions, sample_accounts)
        assert len(scored) == len(sample_transactions)
        assert scored["risk_score"].notna().all()

    def test_scores_within_bounds(self, sample_transactions, sample_accounts):
        scored = score_transactions(sample_transactions, sample_accounts)
        assert (scored["risk_score"] >= 0).all()
        assert (scored["risk_score"] <= 100).all()

    def test_labels_are_valid_values(self, sample_transactions, sample_accounts):
        scored = score_transactions(sample_transactions, sample_accounts)
        assert set(scored["risk_label"]).issubset({"low", "medium", "high"})

    def test_high_risk_transaction_scores_high(self, sample_transactions, sample_accounts):
        scored = score_transactions(sample_transactions, sample_accounts)
        tx2 = scored[scored["transaction_id"] == 2].iloc[0]
        assert tx2["risk_label"] == "high"

    def test_clean_transaction_scores_low(self, sample_transactions, sample_accounts):
        scored = score_transactions(sample_transactions, sample_accounts)
        tx1 = scored[scored["transaction_id"] == 1].iloc[0]
        assert tx1["risk_label"] == "low"

    def test_label_consistent_with_score(self, sample_transactions, sample_accounts):
        scored = score_transactions(sample_transactions, sample_accounts)
        for _, row in scored.iterrows():
            s = row["risk_score"]
            expected = "high" if s >= 60 else "medium" if s >= 30 else "low"
            assert row["risk_label"] == expected, (
                f"transaction {row['transaction_id']}: score {s} labelled {row['risk_label']}"
            )


# ---------------------------------------------------------------------------
# summarize_results
# ---------------------------------------------------------------------------

class TestSummarizeResults:
    def _scored(self, sample_transactions, sample_accounts):
        return score_transactions(sample_transactions, sample_accounts)

    def test_output_has_expected_columns(
        self, sample_transactions, sample_accounts, sample_chargebacks
    ):
        scored = self._scored(sample_transactions, sample_accounts)
        summary = summarize_results(scored, sample_chargebacks)
        for col in ("risk_label", "transactions", "total_amount_usd",
                    "avg_amount_usd", "chargebacks", "chargeback_rate"):
            assert col in summary.columns, f"missing column: {col}"

    def test_transaction_counts_sum_to_total(
        self, sample_transactions, sample_accounts, sample_chargebacks
    ):
        scored = self._scored(sample_transactions, sample_accounts)
        summary = summarize_results(scored, sample_chargebacks)
        assert summary["transactions"].sum() == len(sample_transactions)

    def test_chargeback_rate_between_zero_and_one(
        self, sample_transactions, sample_accounts, sample_chargebacks
    ):
        scored = self._scored(sample_transactions, sample_accounts)
        summary = summarize_results(scored, sample_chargebacks)
        assert (summary["chargeback_rate"] >= 0).all()
        assert (summary["chargeback_rate"] <= 1).all()

    def test_chargeback_matched_to_correct_label(
        self, sample_transactions, sample_accounts, sample_chargebacks
    ):
        scored = self._scored(sample_transactions, sample_accounts)
        summary = summarize_results(scored, sample_chargebacks)
        # TX 2 is the only chargeback and it should be in the "high" bucket
        high_row = summary[summary["risk_label"] == "high"]
        assert len(high_row) == 1
        assert high_row.iloc[0]["chargebacks"] == 1

    def test_no_false_chargebacks_in_low_bucket(
        self, sample_transactions, sample_accounts, sample_chargebacks
    ):
        scored = self._scored(sample_transactions, sample_accounts)
        summary = summarize_results(scored, sample_chargebacks)
        low_row = summary[summary["risk_label"] == "low"]
        if len(low_row):
            assert low_row.iloc[0]["chargebacks"] == 0

    def test_empty_chargebacks_gives_zero_rate(
        self, sample_transactions, sample_accounts
    ):
        scored = self._scored(sample_transactions, sample_accounts)
        empty_cb = pd.DataFrame(columns=["transaction_id"])
        summary = summarize_results(scored, empty_cb)
        assert (summary["chargebacks"] == 0).all()
        assert (summary["chargeback_rate"] == 0).all()
