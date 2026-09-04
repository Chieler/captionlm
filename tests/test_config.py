from captionlm.config import LONG_FORM_DURATION_S, select_cb_weight


def test_select_cb_weight_under_threshold_keeps_base():
    assert select_cb_weight(60.0, base_weight=3.0) == 3.0


def test_select_cb_weight_at_threshold_keeps_base():
    assert select_cb_weight(LONG_FORM_DURATION_S, base_weight=3.0) == 3.0


def test_select_cb_weight_over_threshold_caps_to_one():
    assert select_cb_weight(LONG_FORM_DURATION_S + 1, base_weight=3.0) == 1.0


def test_select_cb_weight_over_threshold_ignores_base():
    assert select_cb_weight(3600.0, base_weight=0.5) == 1.0
