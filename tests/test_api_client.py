import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import ClientSession
import base64

from custom_components.qiara.api.client import (
    QiaraApiClient,
    _generate_hmac_response,
    _hmac_key_candidates,
)
from homeassistant.exceptions import ConfigEntryAuthFailed


@pytest.mark.asyncio
async def test_async_get_sensors_success():
    session = AsyncMock(spec=ClientSession)
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.reason = "OK"
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.read.return_value = b'{"hash_i": 123, "list": [{"id": 1, "type_name": "door", "discarded": 0, "hash_t": 456}]}'

    session.request.return_value.__aenter__.return_value = mock_response

    client = QiaraApiClient(session, 1, "test_secret_key", "https://api.test")
    client._session_id = "pre_set_session"  # bypass auth

    sensors = await client.async_get_sensors()

    assert len(sensors) == 1
    assert sensors[0]["id"] == 1


@pytest.mark.asyncio
async def test_async_get_sensors_auth_fail():
    session = AsyncMock(spec=ClientSession)
    mock_response = AsyncMock()
    mock_response.status = 401
    mock_response.reason = "Unauthorized"

    session.request.return_value.__aenter__.return_value = mock_response

    client = QiaraApiClient(session, 1, "test_secret_key", "https://api.test")
    client._session_id = "expired_session"

    # Re-auth also fails (persistent 401)
    with patch.object(client, "_authenticate", side_effect=ConfigEntryAuthFailed("Auth failed")):
        with pytest.raises(ConfigEntryAuthFailed):
            await client.async_get_sensors()


@pytest.mark.asyncio
async def test_retry_logic():
    session = AsyncMock(spec=ClientSession)

    mock_500 = AsyncMock()
    mock_500.status = 500
    mock_500.reason = "Internal Server Error"

    mock_200 = AsyncMock()
    mock_200.status = 200
    mock_200.reason = "OK"
    mock_200.headers = {"Content-Type": "application/json"}
    mock_200.read.return_value = b'{"hash_i": 123, "list": []}'

    session.request.return_value.__aenter__.side_effect = [mock_500, mock_200]

    client = QiaraApiClient(session, 1, "test_secret_key", "https://api.test")
    client._session_id = "pre_set_session"  # bypass auth

    with patch("asyncio.sleep", return_value=None):
        sensors = await client.async_get_sensors()
        assert sensors == []
        assert session.request.call_count == 2


@pytest.mark.asyncio
async def test_authenticate():
    session = AsyncMock(spec=ClientSession)

    challenge_resp = AsyncMock()
    challenge_resp.status = 200
    challenge_resp.reason = "OK"
    challenge_resp.headers = {}
    challenge_resp.read.return_value = b'{"challenge": "dGVzdA=="}'  # base64("test")

    session_resp = AsyncMock()
    session_resp.status = 200
    session_resp.reason = "OK"
    session_resp.headers = {}
    session_resp.read.return_value = b'{"session": "abc123"}'

    session.request.return_value.__aenter__.side_effect = [challenge_resp, session_resp]

    client = QiaraApiClient(session, 42, "dGVzdA==", "https://api.test")
    await client._authenticate()

    assert client._session_id == "abc123"


def _json_response(status: int, body: bytes) -> AsyncMock:
    resp = AsyncMock()
    resp.status = status
    resp.reason = "OK" if status == 200 else "Bad Request"
    resp.headers = {}
    resp.read.return_value = body
    return resp


def test_hmac_key_candidates_text_first():
    """The token as text comes first, its URL-safe base64 decoding second."""
    token = "ab-_cd"  # URL-safe base64, no padding
    assert _hmac_key_candidates(token) == [b"ab-_cd", base64.urlsafe_b64decode("ab-_cd==")]


def test_hmac_key_candidates_not_base64():
    """A secret that is not base64 only yields its text form."""
    assert _hmac_key_candidates("a") == [b"a"]


@pytest.mark.asyncio
async def test_authenticate_falls_back_to_decoded_key():
    """A 'bad challenge response' on the text key retries with the decoded key."""
    session = AsyncMock(spec=ClientSession)
    challenge = _json_response(200, b'{"challenge": "c"}')
    rejected = _json_response(400, b'{"message": "bad challenge response", "reason": 4}')
    accepted = _json_response(200, b'{"session": "s1"}')
    session.request.return_value.__aenter__.side_effect = [challenge, rejected, challenge, accepted]

    client = QiaraApiClient(session, 42, "dGVzdA==", "https://api.test")
    await client._authenticate()

    assert client._session_id == "s1"
    assert client._hmac_key == b"test"
    sent = [c.kwargs["json"]["challenge_rsp"] for c in session.request.call_args_list if c.kwargs.get("json")]
    assert sent == [_generate_hmac_response("c", b"dGVzdA=="), _generate_hmac_response("c", b"test")]


@pytest.mark.asyncio
async def test_authenticate_rejected_raises_auth_failed():
    """When every key is rejected, auth fails with the device's message."""
    session = AsyncMock(spec=ClientSession)
    challenge = _json_response(200, b'{"challenge": "c"}')
    rejected = _json_response(400, b'{"message": "bad challenge response", "reason": 4}')
    session.request.return_value.__aenter__.side_effect = [challenge, rejected, challenge, rejected]

    client = QiaraApiClient(session, 42, "dGVzdA==", "https://api.test")
    with pytest.raises(ConfigEntryAuthFailed, match="bad challenge response"):
        await client._authenticate()
    assert client._session_id is None
