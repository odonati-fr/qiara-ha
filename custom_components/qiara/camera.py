"""Camera platform for Qiara integration."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hmac
import logging
import re
import secrets
import time
from urllib.parse import urlsplit

from aiohttp import web
from homeassistant.components.camera import Camera, CameraEntityFeature
from homeassistant.components.http import HomeAssistantView
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.network import get_url
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import QiaraDataUpdateCoordinator
from .models import QiaraSensorData

_LOGGER = logging.getLogger(__name__)

# hass.data[DOMAIN] key holding the singleton stream-proxy view.
_CAMERA_PROXY_KEY = "_camera_stream_proxy"

# Snapshot cache TTL — avoids re-opening the shutter on every thumbnail poll.
_SNAPSHOT_CACHE_SECONDS = 10

# Grace after opening the shutter before the camera produces video; without it
# HA's first SRT connect races the camera startup and errors ("Invalid data").
_SHUTTER_WARMUP_SECONDS = 3

# SRT low-latency live path. The hub runs an SRT listener on UDP 9000 reachable
# at the hub's own IP; streamid = stream name and passphrase = /home/open_stream
# both come from the local API. HA's own PyAV can't open SRT, so we register the
# stream with HA's bundled go2rtc (which restreams it via ffmpeg) and hand HA the
# resulting local RTSP URL — smooth, since it's local and SRT keeps up with the
# live edge (unlike the remote HLS path, which stutters as segments expire).
OPT_USE_SRT = "use_srt"
OPT_SRT_PORT = "srt_port"
OPT_SRT_LATENCY = "srt_latency"
DEFAULT_SRT_PORT = 9000

# HA's managed go2rtc: its config lives under hass.data["go2rtc"] (a UnixConnector
# session + placeholder URL) and it serves RTSP on this fixed local port.
GO2RTC_DATA_KEY = "go2rtc"
GO2RTC_RTSP_PORT = 18554

# Set while HA probes stream_source() to pick a WebRTC provider (entity added,
# provider registered/removed). HA only needs the URL scheme then, so the probe
# must not open the shutter: on every HA restart it would otherwise cycle the
# shutter open for ~a minute with nobody watching. A ContextVar (not an
# attribute) so a real viewer request running concurrently isn't mistaken for
# a probe.
_PROBING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "qiara_camera_probing", default=False
)

# HLS playlist/segment names and quality directories proxied to the device.
_SAFE_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9_\-][A-Za-z0-9_.\-]*$")


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Qiara camera entities from config entry."""
    coordinator: QiaraDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]

    # Register the stream proxy view once (it resolves the active coordinator at
    # request time, so it survives reloads and is shared across entries). Cancel
    # its idle timers when this entry unloads.
    proxy: QiaraCameraStreamProxy | None = hass.data[DOMAIN].get(_CAMERA_PROXY_KEY)
    if proxy is None:
        proxy = QiaraCameraStreamProxy()
        hass.http.register_view(proxy)
        hass.data[DOMAIN][_CAMERA_PROXY_KEY] = proxy
    entry.async_on_unload(lambda: proxy.shutdown(entry.entry_id))

    # Older versions registered the camera with has_entity_name=True plus a name
    # equal to the device name, producing a doubled friendly name ("Qiara
    # Camera 1 Qiara Camera 1"). That flag is sticky in the registry, so drop
    # the stale entry and let it re-register cleanly. New entries register with
    # has_entity_name=False and won't match this.
    ent_reg = er.async_get(hass)
    for reg_entry in er.async_entries_for_config_entry(ent_reg, entry.entry_id):
        if reg_entry.domain == "camera" and reg_entry.has_entity_name:
            ent_reg.async_remove(reg_entry.entity_id)
    _migrate_camera_ids(hass, entry)

    cameras = [
        QiaraCamera(coordinator, entry, sensor_id)
        for sensor_id, data in coordinator.data.items()
        if isinstance(data, QiaraSensorData) and data.type == "camera"
    ]
    async_add_entities(cameras)


def _migrate_camera_ids(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Scope camera entity and device ids by entry.

    They used to be the bare node id ("qiara_camera_3", device "3"), which
    collides as soon as several entries are set up: every slave camera reports
    itself as node 1. Keep the entity (history, customisations) and its device
    by renaming their ids in place.
    """
    ent_reg = er.async_get(hass)
    dev_reg = dr.async_get(hass)
    prefix = "qiara_camera_"
    for reg_entry in er.async_entries_for_config_entry(ent_reg, entry.entry_id):
        if reg_entry.domain != "camera" or not reg_entry.unique_id.startswith(prefix):
            continue
        node_id = reg_entry.unique_id.removeprefix(prefix)
        ent_reg.async_update_entity(
            reg_entry.entity_id, new_unique_id=camera_unique_id(entry, node_id)
        )
        for device in dr.async_entries_for_config_entry(dev_reg, entry.entry_id):
            if (DOMAIN, node_id) in device.identifiers:
                dev_reg.async_update_device(
                    device.id, new_identifiers={(DOMAIN, camera_device_key(entry, node_id))}
                )


def camera_unique_id(entry: ConfigEntry, node_id: str) -> str:
    return f"{entry.entry_id}_camera_{node_id}"


def camera_device_key(entry: ConfigEntry, node_id: str) -> str:
    """Device identifier, scoped by entry like the sensor platforms'."""
    return f"{entry.entry_id}_{node_id}"


def build_srt_url(
    host: str,
    port: int | str,
    streamid: str,
    passphrase: str,
    latency: int = 500000,
) -> str:
    """Build the caller-mode SRT URL for the hub's low-latency live stream.

    The hub runs an SRT listener (srt-daemon) on UDP 9000; the client connects
    as caller with the stream name as streamid and the passphrase returned by
    /home/open_stream. latency is in microseconds (500ms; a large buffer delays
    the first frame past PyAV's probe window and HA fails to open the stream).
    """
    params = (
        f"streamid={streamid}&passphrase={passphrase}"
        f"&mode=caller&transtype=live&latency={latency}"
    )
    return f"srt://{host}:{port}?{params}"


class QiaraCameraStreamProxy(HomeAssistantView):
    """Proxy Qiara HLS stream requests, injecting the session auth header.

    Two URL patterns cover the full HLS tree:
      /api/qiara/camera/{token}/{entry_id}/{node_id}/stream/{file}          ← master playlist
      /api/qiara/camera/{token}/{entry_id}/{node_id}/stream/{quality}/{file} ← quality playlist + segments

    The entry id picks the device: each camera of a system (master or slave)
    is its own entry, and slaves all report themselves as the same node id.

    The stream consumer (HA's stream worker or go2rtc) can't send HA auth
    headers, so the view can't require HA auth. Instead every URL carries a
    random per-run token in its path — in the path rather than the query string
    so relative segment URIs in the playlists keep it.
    """

    url = "/api/qiara/camera/{token}/{entry_id}/{node_id}/stream/{file}"
    extra_urls = ["/api/qiara/camera/{token}/{entry_id}/{node_id}/stream/{quality}/{file}"]
    name = "api:qiara:camera:stream"
    requires_auth = False
    cors_allowed = True

    _IDLE_TIMEOUT = 20  # seconds after last segment request before closing the shutter
    _CHUNK = 64 * 1024

    def __init__(self) -> None:
        self.token = secrets.token_urlsafe(32)
        # Keyed by (entry_id, node_id).
        self._shutter_open: set[tuple[str, int]] = set()
        self._idle_tasks: dict[tuple[str, int], asyncio.Task] = {}

    @staticmethod
    def _get_api(hass: HomeAssistant, entry_id: str):
        """Return the API of the entry's coordinator, or None if not loaded."""
        coordinator = hass.data.get(DOMAIN, {}).get(entry_id)
        if isinstance(coordinator, QiaraDataUpdateCoordinator):
            return coordinator.api
        return None

    async def get(
        self,
        request: web.Request,
        token: str,
        entry_id: str,
        node_id: str,
        file: str,
        quality: str | None = None,
    ) -> web.StreamResponse:
        """Proxy a stream request to the device, activating the shutter on first access."""
        if not hmac.compare_digest(token, self.token):
            return web.Response(status=404)
        if not all(
            _SAFE_PATH_SEGMENT.match(part) and ".." not in part
            for part in (file, quality)
            if part is not None
        ):
            return web.Response(status=404)

        hass = request.app["hass"]
        api = self._get_api(hass, entry_id)
        if api is None:
            return web.Response(status=503, text="Qiara integration not loaded")

        # Only the discovered camera may be streamed (and have its shutter opened).
        if not node_id.isdigit() or int(node_id) != api.get_camera_node_id():
            return web.Response(status=404)

        session = async_get_clientsession(hass, verify_ssl=False)
        key = (entry_id, int(node_id))

        # Activate shutter once per camera (idempotent on the device)
        if key not in self._shutter_open:
            try:
                await api.async_activate_camera(key[1])
                self._shutter_open.add(key)
                _LOGGER.debug("Qiara: camera %s shutter opened via proxy", node_id)
            except Exception as err:
                _LOGGER.warning("Qiara: could not activate camera shutter: %s", err)

        # Reset the idle timer — shutter closes if no segment arrives within _IDLE_TIMEOUT
        self._reset_idle_timer(hass, api, key)

        tail = f"{quality}/{file}" if quality else file
        upstream_url = f"{api._base_url}/home/stream/{tail}"

        try:
            session_id = await api.async_ensure_session()
            async with session.get(
                upstream_url,
                headers={"x-hlcore-session-id": session_id},
                ssl=False,
            ) as resp:
                # Stream the upstream body through in chunks rather than buffering
                # the whole segment in memory — lower memory use and latency.
                downstream = web.StreamResponse(
                    status=resp.status,
                    headers={"Access-Control-Allow-Origin": "*"},
                )
                downstream.content_type = resp.content_type or "application/octet-stream"
                await downstream.prepare(request)
                async for chunk in resp.content.iter_chunked(self._CHUNK):
                    await downstream.write(chunk)
                await downstream.write_eof()
                return downstream
        except Exception as err:
            _LOGGER.warning("Qiara stream proxy error for %s: %s", tail, err)
            return web.Response(status=502, text=str(err))

    def _reset_idle_timer(self, hass: HomeAssistant, api, key: tuple[str, int]) -> None:
        """Cancel any existing idle timer and start a fresh one."""
        existing = self._idle_tasks.pop(key, None)
        if existing:
            existing.cancel()
        # Background task so HA cancels it automatically at shutdown instead of
        # warning that a non-critical task is still running.
        self._idle_tasks[key] = hass.async_create_background_task(
            self._close_after_idle(api, key),
            name=f"qiara_camera_{key[0]}_{key[1]}_idle_close",
        )

    async def _close_after_idle(self, api, key: tuple[str, int]) -> None:
        """Close the shutter after the idle timeout expires."""
        try:
            await asyncio.sleep(self._IDLE_TIMEOUT)
        except asyncio.CancelledError:
            return
        _LOGGER.debug("Qiara: camera %s stream idle — closing shutter", key[1])
        try:
            await api.async_deactivate_camera(key[1])
        except Exception as err:
            _LOGGER.warning("Qiara: failed to close shutter after idle: %s", err)
        finally:
            self._shutter_open.discard(key)
            self._idle_tasks.pop(key, None)

    def shutdown(self, entry_id: str) -> None:
        """Cancel the entry's outstanding idle timers (called when it unloads)."""
        for key in [k for k in self._idle_tasks if k[0] == entry_id]:
            self._idle_tasks.pop(key).cancel()
        self._shutter_open = {k for k in self._shutter_open if k[0] != entry_id}


class QiaraCamera(CoordinatorEntity, Camera):
    """Qiara camera entity providing HLS streaming via a local auth proxy."""

    _attr_supported_features = CameraEntityFeature.STREAM

    def __init__(
        self,
        coordinator: QiaraDataUpdateCoordinator,
        entry: ConfigEntry,
        sensor_id: str,
    ) -> None:
        """Initialize the camera entity."""
        CoordinatorEntity.__init__(self, coordinator)
        Camera.__init__(self)
        self._sensor_id = sensor_id
        self._entry_id = entry.entry_id
        self._device_key = camera_device_key(entry, sensor_id)
        self._attr_unique_id = camera_unique_id(entry, sensor_id)
        self._last_image: bytes | None = None
        self._last_image_ts: float = 0.0
        self._shutter_task: asyncio.Task | None = None
        # Each viewing session that opens the shutter bumps the generation; a
        # watcher only closes the shutter if no newer session has opened it
        # since, and the lock keeps open/close writes in order on the device.
        self._shutter_gen = 0
        self._shutter_lock = asyncio.Lock()

        # SRT low-latency live path (see build_srt_url). Enabled by default and
        # falls back to HLS if it can't be built; options can disable it or
        # tune the port/latency.
        opts = entry.options
        self._use_srt: bool = bool(opts.get(OPT_USE_SRT, True))
        self._srt_port = opts.get(OPT_SRT_PORT, DEFAULT_SRT_PORT)
        # 500ms: the hub's SRT peerlatency floor. A large latency (e.g. 3s)
        # delays the first frame past PyAV's probe window and HA fails to open
        # the stream ("Invalid data found"); 500ms opens reliably on the LAN.
        self._srt_latency = int(opts.get(OPT_SRT_LATENCY, 500000))

    @property
    def _sensor(self) -> QiaraSensorData | None:
        data = self.coordinator.data.get(self._sensor_id)
        return data if isinstance(data, QiaraSensorData) else None

    @property
    def name(self) -> str:
        sensor = self._sensor
        return sensor.name if sensor else f"Camera {self._sensor_id}"

    @property
    def available(self) -> bool:
        sensor = self._sensor
        return sensor is not None and sensor.available

    @property
    def device_info(self) -> dict:
        return {
            "identifiers": {(DOMAIN, self._device_key)},
            "name": self.name,
            "manufacturer": "Qiara",
        }

    async def async_refresh_providers(self, *, write_state: bool = True) -> None:
        """Re-pick the WebRTC provider without touching the shutter.

        HA calls stream_source() from here only to learn the URL scheme (on
        entity add, i.e. every HA start or reload, and whenever a WebRTC
        provider such as go2rtc registers), so flag those calls as probes.
        """
        token = _PROBING.set(True)
        try:
            await super().async_refresh_providers(write_state=write_state)
        finally:
            _PROBING.reset(token)

    async def async_create_stream(self):
        """Create HA's stream, reopening the shutter if a cached one is reused.

        HA calls stream_source() only when it first builds the Stream and then
        reuses it for later viewings, so a reused SRT/RTSP stream would play
        with the shutter closed. Re-run the SRT setup to reopen it.
        """
        had_stream = self.stream is not None
        stream = await super().async_create_stream()
        if (
            had_stream
            and stream is not None
            and not _PROBING.get()
            and self._use_srt
            and str(stream.source).startswith("rtsp://")
        ):
            await self._async_srt_stream_source()
        return stream

    async def stream_source(self) -> str | None:
        """Return the live stream URL — SRT if enabled/available, else HLS."""
        if _PROBING.get():
            return await self._async_probe_stream_source()
        if self._use_srt:
            srt_url = await self._async_srt_stream_source()
            if srt_url:
                return srt_url
        return await self._async_hls_stream_source()

    async def _async_probe_stream_source(self) -> str | None:
        """Return the stream URL HA would get, with no device side effects."""
        camera_node_id = self.coordinator.api.get_camera_node_id()
        if camera_node_id is None:
            return None
        if self._use_srt and self._go2rtc() is not None:
            return self._go2rtc_rtsp_url(camera_node_id)
        return await self._async_hls_stream_source()

    def _go2rtc(self) -> tuple[object, str] | None:
        """Return HA's managed go2rtc (session, url), or None if unavailable."""
        go2rtc = self.hass.data.get(GO2RTC_DATA_KEY)
        session = getattr(go2rtc, "session", None)
        url = getattr(go2rtc, "url", None)
        if not (session and url):
            return None
        return session, url

    def _go2rtc_stream_name(self, camera_node_id: int) -> str:
        # Scoped by entry: every slave camera reports the same node id.
        return f"qiara_camera_{self._entry_id}_{camera_node_id}"

    def _go2rtc_rtsp_url(self, camera_node_id: int) -> str:
        return f"rtsp://127.0.0.1:{GO2RTC_RTSP_PORT}/{self._go2rtc_stream_name(camera_node_id)}"

    async def _async_hls_stream_source(self) -> str | None:
        """Return the local proxy URL for the HLS stream."""
        api = self.coordinator.api
        camera_node_id = api.get_camera_node_id()
        if camera_node_id is None:
            return None

        proxy: QiaraCameraStreamProxy | None = self.hass.data.get(DOMAIN, {}).get(
            _CAMERA_PROXY_KEY
        )
        if proxy is None:
            return None

        stream_name = (
            api.get_camera_stream_name()
            or await api.async_fetch_camera_stream_name(camera_node_id)
        )
        if not stream_name:
            _LOGGER.warning("Qiara camera: could not retrieve stream name")
            return None

        ha_url = get_url(self.hass, allow_internal=True)
        return (
            f"{ha_url}/api/qiara/camera/{proxy.token}/{self._entry_id}/{camera_node_id}"
            f"/stream/{stream_name}.m3u8"
        )

    async def _async_srt_stream_source(self) -> str | None:
        """Register the hub's SRT stream with go2rtc and return its RTSP URL.

        HA's PyAV can't open SRT directly, so go2rtc restreams it (via ffmpeg)
        and HA reads the resulting local RTSP. Everything comes from the local
        API: SRT listener on the hub's own host:9000, streamid = stream name,
        passphrase = /home/open_stream. Returns None (fall back to HLS) if any
        piece — go2rtc, streamid, or credentials — is unavailable.

        Only called for a real viewer: it opens the shutter.
        """
        go2rtc = self._go2rtc()
        if go2rtc is None:
            _LOGGER.debug("Qiara SRT: go2rtc unavailable — falling back to HLS")
            return None
        session, url = go2rtc

        api = self.coordinator.api
        camera_node_id = api.get_camera_node_id()
        if camera_node_id is None:
            return None
        streamid = (
            api.get_camera_stream_name()
            or await api.async_fetch_camera_stream_name(camera_node_id)
        )
        host = urlsplit(api._base_url).hostname
        if not (streamid and host):
            _LOGGER.debug("Qiara SRT: missing streamid/host — falling back to HLS")
            return None

        creds = await api.async_open_stream(force=True)
        passphrase = creds.get("passphrase")
        if not passphrase:
            _LOGGER.debug("Qiara SRT: no passphrase from open_stream — falling back to HLS")
            return None

        # SRT only carries video while the camera is capturing, so open the
        # shutter now; the watcher below closes it again when viewing stops.
        # The camera needs a moment after the shutter opens before it produces
        # video, so wait briefly — otherwise HA's first connect races the
        # startup and fails with "Invalid data found" before retrying.
        gen = await self._async_open_shutter(camera_node_id)
        await asyncio.sleep(_SHUTTER_WARMUP_SECONDS)

        srt_url = build_srt_url(host, self._srt_port, streamid, passphrase, self._srt_latency)
        # go2rtc pulls SRT via ffmpeg; copy the HEVC video (no transcode) and
        # remux audio to AAC (raw AAC-in-RTSP copy is rejected by the muxer).
        source = f"ffmpeg:{srt_url}#video=copy#audio=aac"
        name = self._go2rtc_stream_name(camera_node_id)
        try:
            from go2rtc_client import Go2RtcRestClient

            client = Go2RtcRestClient(session, url)
            await client.streams.add(name, source)
        except Exception as err:  # noqa: BLE001 - degrade to HLS on any failure
            _LOGGER.warning(
                "Qiara SRT: could not register go2rtc stream, falling back to HLS: %s", err
            )
            # Nothing will watch this stream, so don't leave the shutter open
            # (the HLS proxy opens it again for its own requests).
            await self._async_close_shutter(camera_node_id, gen)
            return None

        # Nothing in the SRT path closes the physical shutter once HA stops
        # viewing (unlike the HLS proxy's idle timer), so watch go2rtc for this
        # stream's consumers and close the shutter when they drop to zero.
        self._start_shutter_watcher(session, url, name, camera_node_id, gen)

        rtsp_url = self._go2rtc_rtsp_url(camera_node_id)
        _LOGGER.debug("Qiara SRT via go2rtc: %s", rtsp_url)
        return rtsp_url

    async def _async_open_shutter(self, camera_node_id: int) -> int:
        """Open the shutter for a new viewing session; return its generation."""
        self._shutter_gen += 1
        gen = self._shutter_gen
        _LOGGER.info("Qiara: opening camera %s shutter for live view", camera_node_id)
        async with self._shutter_lock:
            try:
                await self.coordinator.api.async_activate_camera(camera_node_id)
            except Exception as err:  # noqa: BLE001 - stream setup continues regardless
                _LOGGER.warning("Qiara: could not open camera %s shutter: %s", camera_node_id, err)
        return gen

    async def _async_close_shutter(self, camera_node_id: int, gen: int) -> None:
        """Close the shutter unless a newer viewing session has reopened it."""
        async with self._shutter_lock:
            if gen != self._shutter_gen:
                return
            try:
                await self.coordinator.api.async_deactivate_camera(camera_node_id)
            except Exception as err:  # noqa: BLE001 - logged; nothing else to do
                _LOGGER.warning("Qiara: could not close camera %s shutter: %s", camera_node_id, err)
                return
        _LOGGER.info("Qiara: closed camera %s shutter (stream ended)", camera_node_id)

    def _start_shutter_watcher(
        self, session, url: str, stream_name: str, camera_node_id: int, gen: int
    ) -> None:
        """Replace any running watcher with one owning this viewing session."""
        if self._shutter_task and not self._shutter_task.done():
            # Superseded watchers never close the shutter (generation check).
            self._shutter_task.cancel()
        self._shutter_task = self.hass.async_create_background_task(
            self._watch_shutter(session, url, stream_name, camera_node_id, gen),
            name=f"qiara_shutter_watch_{stream_name}",
        )

    async def _watch_shutter(
        self, session, url: str, stream_name: str, camera_node_id: int, gen: int
    ) -> None:
        """Close the camera shutter once go2rtc has no consumers for the stream.

        Polls go2rtc's raw /api/streams (the typed client drops the consumer
        list). Waits for a consumer to appear first (stream startup), then
        closes the shutter after it has been idle for a short grace period — or
        gives up and closes it if the stream never started. Also closes it if
        cancelled (entity removed), unless a newer session owns the shutter.
        """
        # Polls are a local HTTP call to go2rtc, only while viewing: keep them
        # short so the shutter closes a few seconds after the last viewer.
        poll, grace, startup = 3, 2, 20
        idle = polls = 0
        seen = False
        streams_url = url.rstrip("/") + "/api/streams"
        try:
            while True:
                await asyncio.sleep(poll)
                polls += 1
                try:
                    async with session.get(streams_url) as resp:
                        info = (await resp.json()).get(stream_name) or {}
                    consumers = len(info.get("consumers") or [])
                except Exception:  # noqa: BLE001 - treat query failure as idle
                    consumers = 0
                if consumers > 0:
                    seen, idle = True, 0
                elif seen:
                    idle += 1
                    if idle >= grace:
                        break
                elif polls >= startup:
                    break  # stream never started; close the shutter anyway
        finally:
            await self._async_close_shutter(camera_node_id, gen)

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return a still image from the camera's live.jpeg snapshot endpoint.

        Cached briefly so repeated thumbnail polls don't re-open the shutter.
        """
        now = time.monotonic()
        if self._last_image and now - self._last_image_ts < _SNAPSHOT_CACHE_SECONDS:
            return self._last_image

        api = self.coordinator.api
        camera_node_id = api.get_camera_node_id()
        if camera_node_id is None:
            return self._last_image

        image = await api.async_get_camera_snapshot(camera_node_id)
        if image:
            self._last_image = image
            self._last_image_ts = now
        return self._last_image

    async def async_turn_on(self) -> None:
        """Open the camera shutter."""
        api = self.coordinator.api
        camera_node_id = api.get_camera_node_id()
        if camera_node_id is not None:
            await api.async_activate_camera(camera_node_id)

    async def async_turn_off(self) -> None:
        """Close the camera shutter."""
        api = self.coordinator.api
        camera_node_id = api.get_camera_node_id()
        if camera_node_id is not None:
            await api.async_deactivate_camera(camera_node_id)

    async def async_will_remove_from_hass(self) -> None:
        """Cancel the shutter watcher and close the shutter on unload."""
        if self._shutter_task and not self._shutter_task.done():
            self._shutter_task.cancel()
        await super().async_will_remove_from_hass()
