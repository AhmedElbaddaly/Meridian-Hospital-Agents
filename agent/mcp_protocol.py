"""
mcp_protocol.py
----------------
A small, dependency-free JSON-RPC 2.0 transport that mirrors the message
shapes used by the Model Context Protocol (initialize, tools/call,
elicitation/create, sampling/createMessage, notifications/*).
"""

import asyncio
import itertools
import json
from typing import Any, Callable, Optional


class JsonRpcEndpoint:
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        request_handler: Optional[Callable[[str, dict], Any]] = None,
        notification_handler: Optional[Callable[[str, dict], Any]] = None,
        name: str = "endpoint",
    ):
        self.reader = reader
        self.writer = writer
        self.request_handler = request_handler
        self.notification_handler = notification_handler
        self.name = name
        self._id_counter = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._closed = False

    async def send_request(self, method: str, params: Optional[dict] = None, timeout: float = 60):
        req_id = next(self._id_counter)
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        fut = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        await self._write(msg)
        return await asyncio.wait_for(fut, timeout=timeout)

    async def send_notification(self, method: str, params: Optional[dict] = None):
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        await self._write(msg)

    async def send_response(self, req_id, result=None, error: Optional[dict] = None):
        msg = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        await self._write(msg)

    async def _write(self, msg: dict):
        line = json.dumps(msg) + "\n"
        self.writer.write(line.encode("utf-8"))
        drain = getattr(self.writer, "drain", None)
        if drain:
            await drain()

    async def run(self):
        while not self._closed:
            line = await self.reader.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            await self._dispatch(msg)

    async def _dispatch(self, msg: dict):
        if "method" in msg and "id" in msg:
            asyncio.create_task(
                self._handle_incoming_request(msg["id"], msg["method"], msg.get("params", {}))
            )
        elif "method" in msg:
            if self.notification_handler:
                asyncio.create_task(self.notification_handler(msg["method"], msg.get("params", {})))
        elif "id" in msg:
            fut = self._pending.pop(msg["id"], None)
            if fut and not fut.done():
                if msg.get("error"):
                    fut.set_exception(RuntimeError(msg["error"].get("message", "RPC error")))
                else:
                    fut.set_result(msg.get("result"))

    async def _handle_incoming_request(self, req_id, method: str, params: dict):
        try:
            result = await self.request_handler(method, params)
            await self.send_response(req_id, result=result)
        except Exception as e:
            await self.send_response(req_id, error={"code": -32000, "message": str(e)})

    def close(self):
        self._closed = True


async def stdio_streams_for_subprocess(proc: asyncio.subprocess.Process):
    return proc.stdout, proc.stdin


async def stdio_streams_for_this_process():
    import sys
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    w_transport, w_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(w_transport, w_protocol, reader, loop)
    return reader, writer
