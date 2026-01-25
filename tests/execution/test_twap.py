from hyprl.execution.algos import TWAPExecutor


def test_twap_nominal() -> None:
    executor = TWAPExecutor(num_slices=4, total_seconds=300)
    slices = executor.build_slices(100.0)
    assert len(slices) == 4
    total_qty = sum(s.qty for s in slices)
    assert abs(total_qty - 100.0) < 1e-6
    delays = [s.delay_sec for s in slices]
    assert delays[0] == 0.0
    assert all(delays[i] <= delays[i + 1] for i in range(len(delays) - 1))
    assert abs(delays[-1] - 300.0) < 1e-6


def test_twap_single_slice_zero_seconds() -> None:
    executor = TWAPExecutor(num_slices=1, total_seconds=0.0)
    slices = executor.build_slices(50.0)
    assert len(slices) == 1
    assert slices[0].qty == 50.0
    assert slices[0].delay_sec == 0.0
