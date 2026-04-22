import pytest
from risk_rules import label_risk, score_transaction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_tx(**overrides):
    """Minimal zero-risk transaction; override individual fields per test."""
    tx = {
        "device_risk_score": 5,
        "is_international": 0,
        "amount_usd": 20.0,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


def score(**overrides):
    return score_transaction(base_tx(**overrides))


# ---------------------------------------------------------------------------
# label_risk — exact bucket boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s,expected", [
    (0,  "low"),
    (29, "low"),
    (30, "medium"),
    (59, "medium"),
    (60, "high"),
    (100, "high"),
])
def test_label_boundaries(s, expected):
    assert label_risk(s) == expected


# ---------------------------------------------------------------------------
# Device risk — exact point contributions and threshold boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device_score,expected_points", [
    (0,  0),   # well below low tier
    (39, 0),   # one below mid-tier threshold
    (40, 10),  # exactly at mid-tier threshold
    (69, 10),  # one below high-tier threshold
    (70, 25),  # exactly at high-tier threshold
    (99, 25),  # well above
])
def test_device_risk_exact_points(device_score, expected_points):
    assert score(device_risk_score=device_score) == expected_points


# ---------------------------------------------------------------------------
# International — exact point contribution
# ---------------------------------------------------------------------------

def test_international_adds_fifteen_points():
    assert score(is_international=1) - score(is_international=0) == 15


def test_domestic_adds_zero_points():
    assert score(is_international=0) == 0


# ---------------------------------------------------------------------------
# Amount — exact point contributions and threshold boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amount,expected_points", [
    (0,      0),
    (499.99, 0),   # one cent below mid-tier
    (500,    10),  # exactly at mid-tier
    (999.99, 10),  # one cent below high-tier
    (1000,   25),  # exactly at high-tier
    (9999,   25),  # well above
])
def test_amount_exact_points(amount, expected_points):
    assert score(amount_usd=amount) == expected_points


# ---------------------------------------------------------------------------
# Velocity — exact point contributions and threshold boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("velocity,expected_points", [
    (1, 0),   # below low threshold
    (2, 0),   # one below mid-tier threshold
    (3, 5),   # exactly at mid-tier threshold
    (5, 5),   # one below high-tier threshold
    (6, 20),  # exactly at high-tier threshold
    (15, 20), # well above
])
def test_velocity_exact_points(velocity, expected_points):
    assert score(velocity_24h=velocity) == expected_points


# ---------------------------------------------------------------------------
# Failed logins — exact point contributions and threshold boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("logins,expected_points", [
    (0, 0),
    (1, 0),   # one below mid-tier threshold
    (2, 10),  # exactly at mid-tier threshold
    (4, 10),  # one below high-tier threshold
    (5, 20),  # exactly at high-tier threshold
    (20, 20), # well above
])
def test_failed_logins_exact_points(logins, expected_points):
    assert score(failed_logins_24h=logins) == expected_points


# ---------------------------------------------------------------------------
# Prior chargebacks — exact point contributions and threshold boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prior_cb,expected_points", [
    (0, 0),
    (1, 5),   # exactly one prior chargeback
    (2, 20),  # exactly at high-tier threshold
    (5, 20),  # well above
])
def test_prior_chargebacks_exact_points(prior_cb, expected_points):
    assert score(prior_chargebacks=prior_cb) == expected_points


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------

def test_score_floor_is_zero():
    # All signals at zero → score must be 0, never negative
    assert score() == 0


def test_score_ceiling_is_100():
    # All signals maxed — raw sum would be 25+15+25+20+20+20 = 125
    assert score(
        device_risk_score=99,
        is_international=1,
        amount_usd=9999,
        velocity_24h=99,
        failed_logins_24h=99,
        prior_chargebacks=99,
    ) == 100


def test_score_additivity_within_cap():
    # With no cap risk: device(10) + intl(15) + amount(10) = 35
    s = score(device_risk_score=50, is_international=1, amount_usd=750)
    assert s == 35


# ---------------------------------------------------------------------------
# Dataset regression — every confirmed chargeback must be flagged
#
# These are the 8 transactions in chargebacks.csv.  Each should score
# at least "medium" after the scoring fixes.  If any regresses to "low",
# real fraud dollars go undetected.
# ---------------------------------------------------------------------------

CONFIRMED_CHARGEBACKS = [
    # (tx_id, device, intl, amount, velocity, logins, prior_cb, expected_min_label)
    (50003, 81, 1, 1250.0, 6,  5, 0, "high"),
    (50006, 77, 1,  399.99, 7,  6, 3, "high"),
    (50008, 68, 1,  620.0,  5,  3, 0, "medium"),
    (50011, 85, 1, 1400.0,  8,  7, 1, "high"),
    (50013, 79, 1,  150.0,  7,  5, 0, "high"),
    (50014, 72, 1,   49.99, 9,  7, 3, "high"),
    (50015, 71, 1,  910.0,  6,  4, 0, "high"),
    (50019, 83, 1,   75.0, 10,  8, 1, "high"),
]

LABEL_ORDER = {"low": 0, "medium": 1, "high": 2}

@pytest.mark.parametrize(
    "tx_id,device,intl,amount,velocity,logins,prior_cb,min_label",
    CONFIRMED_CHARGEBACKS,
    ids=[f"tx{row[0]}" for row in CONFIRMED_CHARGEBACKS],
)
def test_confirmed_chargeback_not_scored_low(
    tx_id, device, intl, amount, velocity, logins, prior_cb, min_label
):
    tx = base_tx(
        device_risk_score=device,
        is_international=intl,
        amount_usd=amount,
        velocity_24h=velocity,
        failed_logins_24h=logins,
        prior_chargebacks=prior_cb,
    )
    result_label = label_risk(score_transaction(tx))
    assert LABEL_ORDER[result_label] >= LABEL_ORDER[min_label], (
        f"TX {tx_id}: expected >= {min_label}, got {result_label}"
    )


# ---------------------------------------------------------------------------
# Clean transaction regression — obvious low-risk txns must not be flagged
# ---------------------------------------------------------------------------

CLEAN_TRANSACTIONS = [
    # (tx_id, device, intl, amount, velocity, logins, prior_cb)
    (50001, 8,  0,  45.20, 1, 0, 0),
    (50009, 6,  0,  18.40, 1, 0, 2),  # prior CB on account but small, domestic, low velocity
    (50012, 10, 0,  64.50, 1, 0, 0),
    (50020, 15, 0, 120.00, 1, 0, 0),
]

@pytest.mark.parametrize(
    "tx_id,device,intl,amount,velocity,logins,prior_cb",
    CLEAN_TRANSACTIONS,
    ids=[f"tx{row[0]}" for row in CLEAN_TRANSACTIONS],
)
def test_clean_transaction_not_scored_high(tx_id, device, intl, amount, velocity, logins, prior_cb):
    tx = base_tx(
        device_risk_score=device,
        is_international=intl,
        amount_usd=amount,
        velocity_24h=velocity,
        failed_logins_24h=logins,
        prior_chargebacks=prior_cb,
    )
    result_label = label_risk(score_transaction(tx))
    assert result_label != "high", (
        f"TX {tx_id}: clean transaction should not be high risk, got {result_label}"
    )
