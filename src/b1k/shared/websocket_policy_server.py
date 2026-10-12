"""Standalone websocket server for the BEHAVIOR evaluator protocol.

This keeps policy serving independent from an OmniGibson installation. The
simulator and policy intentionally run in separate Python environments; only
their msgpack websocket protocol is shared.
"""

import asyncio
from copy import deepcopy
import http
import logging
import time
import traceback
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
import torch
import websockets

try:
    import websockets.asyncio.server as _server
except ImportError:  # pragma: no cover - compatibility with websockets < 13
    import websockets.server as _server


logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serve an ``act``-style policy using BEHAVIOR's websocket protocol."""

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        logger.info("Starting websocket server on %s:%s", self._host, self._port)
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        previous_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                request = msgpack_numpy.unpackb(await websocket.recv(), strict_map_key=False)

                # The evaluator does not wait for a response to reset messages.
                if "reset" in request:
                    self._policy.reset()
                    continue

                infer_start = time.monotonic()
                action = self._policy.act(deepcopy(request))
                infer_time = time.monotonic() - infer_start

                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()
                else:
                    action = np.asarray(action)

                response = {
                    "action": action,
                    "server_timing": {"infer_ms": infer_time * 1000},
                }
                if previous_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total_time * 1000

                await websocket.send(packer.pack(response))
                previous_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                error = traceback.format_exc()
                logger.error("Policy request failed:\n%s", error)
                try:
                    await websocket.send(error)
                    await websocket.close(code=1011, reason="Internal server error")
                finally:
                    raise


def _health_check(connection, request):
    if getattr(request, "path", None) != "/healthz":
        return None
    if hasattr(connection, "respond"):
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
