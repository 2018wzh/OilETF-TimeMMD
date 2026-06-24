import pandas as pd

from src.collect.intraday_prices import normalize_yfinance_hour_bars


def test_normalize_yfinance_hour_bars_filters_regular_session_and_shortens_last_bar():
    idx = pd.DatetimeIndex(
        [
            "2026-05-29 08:30:00-04:00",
            "2026-05-29 09:30:00-04:00",
            "2026-05-29 10:30:00-04:00",
            "2026-05-29 15:30:00-04:00",
        ],
        name="Datetime",
    )
    frame = pd.DataFrame(
        {
            ("Open", "USO"): [1.0, 10.0, 11.0, 15.0],
            ("High", "USO"): [1.0, 12.0, 13.0, 16.0],
            ("Low", "USO"): [1.0, 9.0, 10.0, 14.5],
            ("Close", "USO"): [1.0, 11.0, 12.0, 15.5],
            ("Volume", "USO"): [1, 100, 200, 300],
            ("Adj Close", "USO"): [1.0, 11.0, 12.0, 15.5],
        },
        index=idx,
    )

    out = normalize_yfinance_hour_bars(frame, "USO")

    assert list(out["bar_start_et"]) == [
        "2026-05-29T09:30:00-04:00",
        "2026-05-29T10:30:00-04:00",
        "2026-05-29T15:30:00-04:00",
    ]
    assert out.iloc[0]["open"] == 10.0
    assert out.iloc[0]["close"] == 11.0
    assert out.iloc[-1]["bar_end_et"] == "2026-05-29T16:00:00-04:00"
    assert out.iloc[-1]["bar_minutes_actual"] == 30
    assert out["provider"].eq("yfinance").all()
