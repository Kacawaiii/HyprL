import pytest

from scripts.social_publisher.core import Blocked
from scripts.social_publisher.media import MediaPool, byte_range


def test_short_lived_url_is_unguessable_digest_bound_and_expires(tmp_path, bundle):
    clock = [1000]
    pool = MediaPool(tmp_path / "media", clock=lambda: clock[0])
    url = pool.issue(bundle, ttl=300)
    path = url.removeprefix("https://hyprlcore.com")
    assert len(path.split("/")[2]) == 64
    assert pool.read(path) == bundle.video
    assert pool.issue(bundle, ttl=300) != url
    with pytest.raises(Blocked):
        pool.read("/social-media/../video.mp4")
    clock[0] = 1300
    with pytest.raises(Blocked, match="expired"):
        pool.read(path)
    pool.prune()
    assert not list(pool.directory.iterdir())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, (0, 99, 200)),
        ("bytes=10-19", (10, 19, 206)),
        ("bytes=-10", (90, 99, 206)),
        ("bytes=99-", (99, 99, 206)),
        ("bytes=0-999", (0, 99, 206)),
    ],
)
def test_media_get_supports_bounded_http_ranges(value, expected):
    assert byte_range(value, 100) == expected


@pytest.mark.parametrize(
    "value", ["bytes=100-", "bytes=5-1", "bytes=0-1,4-5", "bytes=-0", "bytes=-", "bad"]
)
def test_invalid_media_ranges_are_refused(value):
    with pytest.raises(Blocked):
        byte_range(value, 100)
