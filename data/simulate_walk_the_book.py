import numpy as np
import math
import warnings

# pytest is only needed for the self-tests; guard the import so runtime use
# (e.g., in older Python environments) does not fail.
try:  # pragma: no cover - optional
    import pytest  # type: ignore
except Exception:  # pragma: no cover
    pytest = None

# Simulate Walk the Book Function, my function handles both buy and sell orders 
# using the convetion that buy orders are positive and sell orders are negative. 

def simulate_walk_the_book(
    positions_np: np.ndarray, ask_prices_np: np.ndarray,
      ask_vols_np: np.ndarray, bid_prices_np: np.ndarray, bid_vols_np: np.ndarray
    ) -> tuple[float, float]:
    """
        Simulates walking the order book to execute trades based on positions.

    Args:
        positions_np:: Array of desired positions at each second.
                                   Positive for buying (consuming ask), negative for selling (consuming bid).
        ask_prices_np: NumPy array of ask prices [seconds, levels].
        ask_vols_np: NumPy array of ask volumes [seconds, levels].
        bid_prices_np: NumPy array of bid prices [seconds, levels].
        bid_vols_np: NumPy array of bid volumes [seconds, levels].

    Returns:
        tuple: (total_volume_executed, average_execution_price)
               Returns (0.0, np.nan) if no volume is executed.
    """

    total_volume_executed = 0.0
    total_value_executed = 0.0
    remaining_position = 0.0

    num_seconds = len(positions_np)
    num_levels = ask_prices_np.shape[1] #assumes ask_prices_np and bid_prices_np have the same shape

    for i in range(num_seconds):
        position_needed = positions_np[i] + remaining_position
        remaining_position = 0.0

        if position_needed > 0:
            position_to_fill = position_needed
            for level in range(num_levels):
                price = ask_prices_np[i, level]
                volume_available = ask_vols_np[i, level]
                if math.isnan(price) or math.isnan(volume_available):
                    warnings.warn(
                        f"NaN price or volume encountered in ASK book at second {i} level {level + 1}. Skipping level.", UserWarning
                    )  
                    continue
                if position_to_fill <= volume_available:
                    total_value_executed += position_to_fill * price
                    total_volume_executed += position_to_fill
                    position_to_fill = 0.0
                    break
                else:
                    total_value_executed += volume_available * price
                    total_volume_executed += volume_available
                    position_to_fill -= volume_available
            remaining_position = position_to_fill
        elif position_needed < 0:
            position_to_fill = -position_needed
            for level in range(num_levels):
                price = bid_prices_np[i, level]
                volume_available = bid_vols_np[i, level]
                if math.isnan(price) or math.isnan(volume_available): 
                    warnings.warn(f"NaN price or volume encountered in BID book at second {i} level {level + 1}. Skipping level.", UserWarning)
                    continue
                if position_to_fill <= volume_available:
                    total_value_executed += position_to_fill * price
                    total_volume_executed += position_to_fill
                    position_to_fill = 0.0
                    break
                else:
                    total_value_executed += volume_available * price
                    total_volume_executed += volume_available
                    position_to_fill -= volume_available
            remaining_position = -position_to_fill

    if total_volume_executed == 0:
        return 0.0, np.nan
    average_price = total_value_executed / total_volume_executed
    if math.isnan(average_price):
        warnings.warn("Warning: Final average price calculation resulted in NaN.", UserWarning)
    return total_volume_executed, average_price

def create_full_book_data(
    ask_prices_list, ask_vols_list, bid_prices_list, bid_vols_list, num_seconds, num_levels
):
    # Ensure arrays are float64 for NaN support and consistency with np.nan
    ap = np.full((num_seconds, num_levels), np.nan, dtype=np.float64)
    av = np.full((num_seconds, num_levels), np.nan, dtype=np.float64)
    bp = np.full((num_seconds, num_levels), np.nan, dtype=np.float64)
    bv = np.full((num_seconds, num_levels), np.nan, dtype=np.float64)

    for s_idx in range(num_seconds):
        for l_idx in range(num_levels):
            # Populate ASK side
            if s_idx < len(ask_prices_list) and l_idx < len(ask_prices_list[s_idx]):
                ap[s_idx, l_idx] = float(ask_prices_list[s_idx][l_idx])
            if s_idx < len(ask_vols_list) and l_idx < len(ask_vols_list[s_idx]):
                av[s_idx, l_idx] = float(ask_vols_list[s_idx][l_idx])

            # Populate BID side
            if s_idx < len(bid_prices_list) and l_idx < len(bid_prices_list[s_idx]):
                bp[s_idx, l_idx] = float(bid_prices_list[s_idx][l_idx])
            if s_idx < len(bid_vols_list) and l_idx < len(bid_vols_list[s_idx]):
                bv[s_idx, l_idx] = float(bid_vols_list[s_idx][l_idx])
    return ap, av, bp, bv


# --- BUY ORDER TESTS ---

def test_buy_simple_fill_first_level():
    positions = np.array([10.0])
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[100.0, 101.0]], ask_vols_list=[[20.0, 30.0]],
        bid_prices_list=[[0.0, 0.0]], bid_vols_list=[[0.0, 0.0]],
        num_seconds=1, num_levels=2
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    assert total_vol == pytest.approx(10.0)
    assert avg_price == pytest.approx(100.0)

def test_buy_fill_multiple_levels():
    positions = np.array([40.0])
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[100.0, 101.0, 102.0]], ask_vols_list=[[15.0, 20.0, 30.0]],
        bid_prices_list=[[0.0, 0.0, 0.0]], bid_vols_list=[[0.0, 0.0, 0.0]],
        num_seconds=1, num_levels=3
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    expected_total_value = (15 * 100) + (20 * 101) + (5 * 102)
    expected_total_volume = 40.0
    expected_avg_price = expected_total_value / expected_total_volume

    assert total_vol == pytest.approx(expected_total_volume)
    assert avg_price == pytest.approx(expected_avg_price)

def test_buy_partial_fill_carry_over():
    positions = np.array([30.0, 10.0]) # Buy 30 in sec 0, Buy 10 in sec 1
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[100.0, 101.0, 102.0], [102.0, 103.0, 104.0]],
        ask_vols_list=[[10.0, 10.0, 0], [10.0, 10.0, 50.0]],
        bid_prices_list=[[0.0]*3, [0.0]*3], bid_vols_list=[[0.0]*3, [0.0]*3],
        num_seconds=2, num_levels=3
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    # Expected:
    # Sec 0 (Buy 30): 10@100 + 10@101 = 20 vol, 10 remaining (internal)
    # Sec 1 (Buy 10 + 10 carried = 20 needed): 10@102 + 10@103 = 20 vol
    # Total volume: 20 + 20 = 40
    # Total value: (10*100 + 10*101) + (10*102 + 10*103) = 2010 + 2050 = 4060
    # Average price: 4060 / 40 = 101.5

    assert total_vol == pytest.approx(40.0)
    assert avg_price == pytest.approx(101.5)

def test_buy_cannot_fill_completely():
    positions = np.array([50.0]) # Try to buy 50
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[100.0, 101.0]], ask_vols_list=[[10.0, 10.0]], # Only 20 vol available
        bid_prices_list=[[0.0, 0.0]], bid_vols_list=[[0.0, 0.0]],
        num_seconds=1, num_levels=2
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    # Expected: 10@100 + 10@101 = 20 filled. 30 remaining internally.
    expected_total_value = (10 * 100) + (10 * 101)
    expected_total_volume = 20.0
    expected_avg_price = expected_total_value / expected_total_volume

    assert total_vol == pytest.approx(expected_total_volume)
    assert avg_price == pytest.approx(expected_avg_price)

def test_buy_zero_liquidity():
    positions = np.array([10.0])
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[100.0, 101.0]], ask_vols_list=[[0.0, 0.0]], # No volume available
        bid_prices_list=[[0.0, 0.0]], bid_vols_list=[[0.0, 0.0]],
        num_seconds=1, num_levels=2
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    assert total_vol == pytest.approx(0.0)
    assert np.isnan(avg_price)

def test_buy_nan_price_skips_level_and_warns():
    positions = np.array([10.0])
    # First level has NaN price
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[np.nan, 101.0]], ask_vols_list=[[5.0, 10.0]],
        bid_prices_list=[[0.0, 0.0]], bid_vols_list=[[0.0, 0.0]],
        num_seconds=1, num_levels=2
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        total_vol, avg_price = simulate_walk_the_book(
            positions, ask_prices, ask_vols, bid_prices, bid_vols
        )
        assert len(w) >= 1
        assert issubclass(w[-1].category, UserWarning)
        assert "NaN price or volume encountered in ASK book" in str(w[-1].message)

    # Expected: 5 units available at first level (NaN price) are skipped.
    # Order for 10 units goes to second level: 10@101
    assert total_vol == pytest.approx(10.0)
    assert avg_price == pytest.approx(101.0)

def test_buy_nan_volume_skips_level_and_warns():
    positions = np.array([10.0])
    # First level has NaN volume
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[100.0, 101.0]], ask_vols_list=[[np.nan, 10.0]],
        bid_prices_list=[[0.0, 0.0]], bid_vols_list=[[0.0, 0.0]],
        num_seconds=1, num_levels=2
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        total_vol, avg_price = simulate_walk_the_book(
            positions, ask_prices, ask_vols, bid_prices, bid_vols
        )
        assert len(w) >= 1
        assert issubclass(w[-1].category, UserWarning)
        assert "NaN price or volume encountered in ASK book" in str(w[-1].message)

    # Expected: 5 units available at first level (NaN volume) are skipped.
    # Order for 10 units goes to second level: 10@101
    assert total_vol == pytest.approx(10.0)
    assert avg_price == pytest.approx(101.0)

# --- SELL ORDER TESTS ---

def test_sell_simple_fill_first_level():
    positions = np.array([-5.0])
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[0.0, 0.0]], ask_vols_list=[[0.0, 0.0]], # Ask not relevant for sell
        bid_prices_list=[[99.0, 98.0]], bid_vols_list=[[20.0, 30.0]],
        num_seconds=1, num_levels=2
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    assert total_vol == pytest.approx(5.0) # Absolute volume
    assert avg_price == pytest.approx(99.0)

def test_sell_fill_multiple_levels():
    positions = np.array([-30.0])
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[0.0, 0.0, 0.0]], ask_vols_list=[[0.0, 0.0, 0.0]],
        bid_prices_list=[[99.0, 98.0, 97.0]], bid_vols_list=[[10.0, 15.0, 20.0]],
        num_seconds=1, num_levels=3
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    expected_total_value = (10 * 99) + (15 * 98) + (5 * 97)
    expected_total_volume = 30.0
    expected_avg_price = expected_total_value / expected_total_volume

    assert total_vol == pytest.approx(expected_total_volume)
    assert avg_price == pytest.approx(expected_avg_price)

def test_sell_partial_fill_carry_over():
    positions = np.array([-25.0, -5.0]) # Sell 25 in sec 0, Sell 5 in sec 1
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[0.0]*3, [0.0]*3], ask_vols_list=[[0.0]*3, [0.0]*3],
        bid_prices_list=[[99.0, 98.0, 97.0], [97.0, 96.0, 95.0]],
        bid_vols_list=[[10.0, 10.0, 50.0], [10.0, 10.0, 50.0]],
        num_seconds=2, num_levels=3
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    # Expected:
    # Sec 0 (Sell 25): 10@99 + 10@98 = 20 vol, 5 remaining (-5.0 for carry-over)
    # Sec 1 (Sell 5 + 5 carried = 10 needed): 10@97 = 10 vol
    # Total volume: 20 + 10 = 30
    # Total value: (10*99 + 10*98) + (10*97) = 1970 + 970 = 2940
    # Average price: 2940 / 30 = 98.0

    assert total_vol == pytest.approx(30.0)
    assert avg_price == pytest.approx(98.0)

def test_sell_cannot_fill_completely():
    positions = np.array([-50.0]) # Try to sell 50
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[0.0, 0.0]], ask_vols_list=[[0.0, 0.0]],
        bid_prices_list=[[99.0, 98.0]], bid_vols_list=[[10.0, 10.0]], # Only 20 vol available
        num_seconds=1, num_levels=2
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    # Expected: 10@99 + 10@98 = 20 filled. 30 remaining internally (-30.0).
    expected_total_value = (10 * 99) + (10 * 98)
    expected_total_volume = 20.0
    expected_avg_price = expected_total_value / expected_total_volume

    assert total_vol == pytest.approx(expected_total_volume)
    assert avg_price == pytest.approx(expected_avg_price)

def test_sell_zero_liquidity():
    positions = np.array([-10.0])
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[0.0, 0.0]], ask_vols_list=[[0.0, 0.0]],
        bid_prices_list=[[99.0, 98.0]], bid_vols_list=[[0.0, 0.0]], # No volume available
        num_seconds=1, num_levels=2
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    assert total_vol == pytest.approx(0.0)
    assert np.isnan(avg_price)

def test_sell_nan_price_skips_level_and_warns():
    positions = np.array([-10.0])
    # First level has NaN price
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[0.0, 0.0]], ask_vols_list=[[0.0, 0.0]],
        bid_prices_list=[[np.nan, 98.0]], bid_vols_list=[[5.0, 10.0]],
        num_seconds=1, num_levels=2
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        total_vol, avg_price = simulate_walk_the_book(
            positions, ask_prices, ask_vols, bid_prices, bid_vols
        )
        assert len(w) >= 1
        assert issubclass(w[-1].category, UserWarning)
        assert "NaN price or volume encountered in BID book" in str(w[-1].message)

    # Expected: 5 units available at first level (NaN price) are skipped.
    # Order for 10 units goes to second level: 10@98
    assert total_vol == pytest.approx(10.0)
    assert avg_price == pytest.approx(98.0)

def test_sell_nan_volume_skips_level_and_warns():
    positions = np.array([-10.0])
    # First level has NaN volume
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[0.0, 0.0]], ask_vols_list=[[0.0, 0.0]],
        bid_prices_list=[[99.0, 98.0]], bid_vols_list=[[np.nan, 10.0]],
        num_seconds=1, num_levels=2
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        total_vol, avg_price = simulate_walk_the_book(
            positions, ask_prices, ask_vols, bid_prices, bid_vols
        )
        assert len(w) >= 1
        assert issubclass(w[-1].category, UserWarning)
        assert "NaN price or volume encountered in BID book" in str(w[-1].message)

    # Expected: 5 units available at first level (NaN volume) are skipped.
    # Order for 10 units goes to second level: 10@98
    assert total_vol == pytest.approx(10.0)
    assert avg_price == pytest.approx(98.0)

# --- GENERAL TESTS (apply to both buy/sell logic via internal remaining_position) ---

def test_zero_position_no_trades():
    positions = np.array([0.0, 0.0])
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[100.0], [100.0]], ask_vols_list=[[10.0], [10.0]],
        bid_prices_list=[[99.0], [99.0]], bid_vols_list=[[10.0], [10.0]],
        num_seconds=2, num_levels=1
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    assert total_vol == pytest.approx(0.0)
    assert np.isnan(avg_price)

def test_empty_positions_array():
    positions = np.array([])
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[], ask_vols_list=[],
        bid_prices_list=[], bid_vols_list=[],
        num_seconds=0, num_levels=1 # num_seconds must be 0 for empty positions
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    assert total_vol == pytest.approx(0.0)
    assert np.isnan(avg_price)


def test_initial_negative_remaining_position_sell_side_only():
    # Simulate initial carry-over for a sell order
    positions = np.array([-5.0])
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[0.0, 0.0]], ask_vols_list=[[0.0, 0.0]],
        bid_prices_list=[[99.0, 98.0]], bid_vols_list=[[10.0, 10.0]],
        num_seconds=1, num_levels=2
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    assert total_vol == pytest.approx(5.0)
    assert avg_price == pytest.approx(99.0)

def test_initial_positive_remaining_position_buy_side_only():
    # Simulate initial carry-over for a buy order
    positions = np.array([5.0])
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[100.0, 101.0]], ask_vols_list=[[10.0, 10.0]],
        bid_prices_list=[[0.0, 0.0]], bid_vols_list=[[0.0, 0.0]],
        num_seconds=1, num_levels=2
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    assert total_vol == pytest.approx(5.0)
    assert avg_price == pytest.approx(100.0)

def test_multi_second_unfilled_at_end_buy_side_only():
    positions = np.array([10.0, 15.0]) # Buy 10, then Buy 15
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[100.0, 101.0], [102.0, 103.0]],
        ask_vols_list=[[5.0, 5.0], [5.0, 5.0]],
        bid_prices_list=[[0.0, 0.0]]*2, bid_vols_list=[[0.0, 0.0]]*2,
        num_seconds=2, num_levels=2
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    assert total_vol == pytest.approx(20.0)
    assert avg_price == pytest.approx(101.5)

def test_multi_second_unfilled_at_end_sell_side_only():
    positions = np.array([-10.0, -15.0]) # Sell 10, then Sell 15
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[0.0, 0.0]]*2, ask_vols_list=[[0.0, 0.0]]*2,
        bid_prices_list=[[99.0, 98.0], [97.0, 96.0]],
        bid_vols_list=[[5.0, 5.0], [5.0, 5.0]],
        num_seconds=2, num_levels=2
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    assert total_vol == pytest.approx(20.0)
    assert avg_price == pytest.approx(97.5)

def test_no_trading_possible_multiple_seconds_buy_side_only():
    positions = np.array([10.0, 20.0, 30.0])
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[100.0, 101.0]]*3, ask_vols_list=[[0.0, 0.0]]*3,
        bid_prices_list=[[0.0, 0.0]]*3, bid_vols_list=[[0.0, 0.0]]*3,
        num_seconds=3, num_levels=2
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    assert total_vol == pytest.approx(0.0)
    assert np.isnan(avg_price)

def test_no_trading_possible_multiple_seconds_sell_side_only():
    positions = np.array([-10.0, -20.0, -30.0])
    ask_prices, ask_vols, bid_prices, bid_vols = create_full_book_data(
        ask_prices_list=[[0.0, 0.0]]*3, ask_vols_list=[[0.0, 0.0]]*3,
        bid_prices_list=[[99.0, 98.0]]*3, bid_vols_list=[[0.0, 0.0]]*3,
        num_seconds=3, num_levels=2
    )

    total_vol, avg_price = simulate_walk_the_book(
        positions, ask_prices, ask_vols, bid_prices, bid_vols
    )
    assert total_vol == pytest.approx(0.0)
    assert np.isnan(avg_price)