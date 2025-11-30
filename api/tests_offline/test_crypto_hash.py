from api.utils.crypto import hash_token, verify_token


def test_hash_and_verify_token() -> None:
    hashed = hash_token("secret")
    assert hashed != "secret"
    assert verify_token("secret", hashed)
    assert not verify_token("oops", hashed)
