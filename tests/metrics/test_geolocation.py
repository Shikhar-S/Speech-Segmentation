import pytest
import torch
import math
from src.metrics.geolocation import GeolocationDistanceError, GeolocationMissRate

# Constants
R_KM = 6378.1


def latlong_to_cartesian(lat_rad, lon_rad):
    """Helper to generate Cartesian inputs for tests."""
    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    return torch.stack([x, y, z], dim=-1)


# -------------------------------------------------------------------------
# 1. GeolocationDistanceError Tests
# -------------------------------------------------------------------------


def test_distance_error_perfect_match():
    """Test that error is 0.0 when predictions exactly match targets."""
    metric = GeolocationDistanceError(earth_radius_km=R_KM)

    # Define targets (Lat, Lon in radians)
    targets = torch.tensor(
        [
            [0.0, 0.0],  # (0, 0)
            [math.pi / 2, 0.0],  # North Pole
            [0.0, math.pi],  # Opposite side of globe
        ]
    )

    # Preds match targets exactly
    preds = latlong_to_cartesian(targets[:, 0], targets[:, 1])

    metric.update(preds, targets)
    result = metric.compute()

    # Now that we clamped properly, this should be very close to 0
    assert torch.isclose(result, torch.tensor(0.0), atol=1e-3)


def test_distance_error_known_values():
    """Test specific known Great Circle distances."""
    metric = GeolocationDistanceError(earth_radius_km=R_KM)

    # Case: Equator (0,0) vs North Pole (pi/2, 0)
    # Distance should be 1/4 of circumference: (2*pi*R) / 4 = pi*R / 2
    target = torch.tensor([[math.pi / 2, 0.0]])  # True is North Pole
    pred_latlon = torch.tensor([[0.0, 0.0]])  # Pred is (0,0)
    pred = latlong_to_cartesian(pred_latlon[:, 0], pred_latlon[:, 1])

    metric.update(pred, target)
    result = metric.compute()

    expected_km = (math.pi / 2) * R_KM
    assert torch.isclose(result, torch.tensor(expected_km), rtol=1e-4)


def test_distance_error_batch_average():
    """Test that the metric correctly averages error over a batch."""
    metric = GeolocationDistanceError(earth_radius_km=R_KM)

    # Sample 1: Error = 0
    t1 = torch.tensor([0.0, 0.0])
    p1 = latlong_to_cartesian(torch.tensor([0.0]), torch.tensor([0.0]))[0]

    # Sample 2: Error = pi/2 radians
    t2 = torch.tensor([math.pi / 2, 0.0])
    p2 = latlong_to_cartesian(torch.tensor([0.0]), torch.tensor([0.0]))[0]  # Equator

    targets = torch.stack([t1, t2])
    preds = torch.stack([p1, p2])

    metric.update(preds, targets)
    result = metric.compute()

    expected = (0.0 + (math.pi / 2 * R_KM)) / 2
    assert torch.isclose(result, torch.tensor(expected), rtol=1e-4)


# -------------------------------------------------------------------------
# 2. GeolocationMissRate Tests
# -------------------------------------------------------------------------


def test_miss_rate_perfect_recall():
    """Test that Miss Rate is 0% when every pred is closest to its own target."""
    metric = GeolocationMissRate(k=1)

    # Three distinct targets
    targets = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    # Perfect predictions
    preds = latlong_to_cartesian(targets[:, 0], targets[:, 1])

    metric.update(preds, targets)
    result = metric.compute()

    # Should be 0.0% miss rate (100% recall)
    assert result.item() == 0.0


def test_miss_rate_all_wrong():
    """Test that Miss Rate is 100% when preds are closer to neighbors."""
    metric = GeolocationMissRate(k=1)

    # Two targets: A and B
    targets = torch.tensor([[0.0, 0.0], [1.0, 0.0]])  # Target A  # Target B

    # Swap predictions:
    # Pred 0 is closer to Target B
    # Pred 1 is closer to Target A
    preds = latlong_to_cartesian(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0]))

    metric.update(preds, targets)
    result = metric.compute()

    # Since k=1 and we swapped them, miss rate should be 100%
    assert result.item() == 100.0


def test_miss_rate_top_k():
    """Test Top-2 Miss Rate. Even if Top-1 is wrong, Top-2 might be right."""
    metric = GeolocationMissRate(k=2)

    targets = torch.tensor(
        [
            [0.0, 0.0],  # T0
            [0.1, 0.0],  # T1 (Very close to T0)
            [1.0, 0.0],  # T2 (Far away)
        ]
    )

    # Construct Preds manually
    p0 = latlong_to_cartesian(torch.tensor(0.08), torch.tensor(0.0))
    p1 = latlong_to_cartesian(torch.tensor(0.1), torch.tensor(0.0))
    p2 = latlong_to_cartesian(torch.tensor(1.0), torch.tensor(0.0))

    # Stack to create (3, 3) tensor
    preds = torch.stack([p0, p1, p2])

    metric.update(preds, targets)
    result = metric.compute()

    # Top-2 logic:
    # P0's neighbors: [T1, T0, T2]. T0 is in top 2. -> Hit.
    # P1's neighbors: [T1, T0, T2]. T1 is in top 2. -> Hit.
    # P2's neighbors: [T2, ...].    T2 is in top 2. -> Hit.
    # Total Miss Rate should be 0.0%
    assert result.item() == 0.0

    # Now Check Top-1 for the same data
    metric_k1 = GeolocationMissRate(k=1)
    metric_k1.update(preds, targets)
    result_k1 = metric_k1.compute()

    # P0 closest to T1 (Actual T0) -> Miss
    # P1 closest to T1 (Actual T1) -> Hit
    # P2 closest to T2 (Actual T2) -> Hit
    # Miss Rate = 1/3 * 100 = 33.33%
    assert torch.isclose(result_k1, torch.tensor(33.3333), atol=1e-3)


def test_empty_update():
    """Ensure it doesn't crash on empty input."""
    metric = GeolocationMissRate(k=1)
    result = metric.compute()
    assert result.item() == 0.0


def test_fewer_samples_than_k():
    """Ensure k=10 doesn't crash if batch size is only 2."""
    metric = GeolocationMissRate(k=10)

    targets = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    preds = latlong_to_cartesian(targets[:, 0], targets[:, 1])

    metric.update(preds, targets)
    result = metric.compute()
    assert result.item() == 0.0


# -------------------------------------------------------------------------
# 3. GPU/Device Compatibility Tests
# -------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_metrics_on_gpu():
    """Ensure metrics run correctly on GPU without device mismatch errors."""
    device = "cuda"

    # Setup data on GPU
    targets = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device)
    preds = latlong_to_cartesian(targets[:, 0], targets[:, 1]).to(device)

    # 1. Distance Error
    metric_dist = GeolocationDistanceError().to(device)
    metric_dist.update(preds, targets)
    res_dist = metric_dist.compute()

    # Assert result is reasonable and didn't crash
    # TorchMetrics usually syncs results back to CPU for safety on compute(),
    # but the internal states should have handled GPU tensors correctly.
    assert res_dist >= 0.0

    # 2. Miss Rate
    metric_miss = GeolocationMissRate(k=1).to(device)
    metric_miss.update(preds, targets)
    res_miss = metric_miss.compute()

    # Since preds match targets, miss rate should be 0
    assert res_miss == 0.0
