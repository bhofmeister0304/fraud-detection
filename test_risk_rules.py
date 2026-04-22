from risk_rules import label_risk, score_transaction


# --- helpers ---

def base_tx(**overrides):
    """Minimal low-risk transaction; override individual fields per test."""
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


# --- label thresholds ---

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


# --- individual signals ---

def test_high_risk_device_adds_risk():
    low_dev = score_transaction(base_tx(device_risk_score=5))
    mid_dev = score_transaction(base_tx(device_risk_score=50))
    high_dev = score_transaction(base_tx(device_risk_score=80))
    assert high_dev > mid_dev > low_dev


def test_international_adds_risk():
    domestic = score_transaction(base_tx(is_international=0))
    international = score_transaction(base_tx(is_international=1))
    assert international > domestic


def test_large_amount_adds_risk():
    small = score_transaction(base_tx(amount_usd=100))
    medium = score_transaction(base_tx(amount_usd=750))
    large = score_transaction(base_tx(amount_usd=1200))
    assert large > medium > small


def test_high_velocity_adds_risk():
    low_vel = score_transaction(base_tx(velocity_24h=1))
    mid_vel = score_transaction(base_tx(velocity_24h=4))
    high_vel = score_transaction(base_tx(velocity_24h=8))
    assert high_vel > mid_vel > low_vel


def test_failed_logins_add_risk():
    no_fail = score_transaction(base_tx(failed_logins_24h=0))
    some_fail = score_transaction(base_tx(failed_logins_24h=3))
    many_fail = score_transaction(base_tx(failed_logins_24h=6))
    assert many_fail > some_fail > no_fail


def test_prior_chargebacks_add_risk():
    no_cb = score_transaction(base_tx(prior_chargebacks=0))
    one_cb = score_transaction(base_tx(prior_chargebacks=1))
    two_cb = score_transaction(base_tx(prior_chargebacks=2))
    assert two_cb > one_cb > no_cb


# --- score bounds ---

def test_score_never_below_zero():
    assert score_transaction(base_tx()) >= 0


def test_score_never_above_100():
    worst = base_tx(
        device_risk_score=90,
        is_international=1,
        amount_usd=5000,
        velocity_24h=10,
        failed_logins_24h=10,
        prior_chargebacks=5,
    )
    assert score_transaction(worst) <= 100


# --- known fraud pattern (TX 50011 from dataset) ---

def test_high_risk_combination_scores_high():
    # RU IP, $1400, device=85, velocity=8, 7 failed logins, 1 prior chargeback
    tx = base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1400,
        velocity_24h=8,
        failed_logins_24h=7,
        prior_chargebacks=1,
    )
    assert label_risk(score_transaction(tx)) == "high"
