#!/usr/bin/env python3
"""Interactive MCP client for Atlassian's remote MCP server.

Connects to https://mcp.atlassian.com over streamable HTTP using OAuth 2.1
(authorization code + PKCE, dynamic client registration). The registered
client and access/refresh tokens are cached to a local JSON file so
re-running the script does not require a fresh browser login every time.

Usage:
    python atlassian_mcp_client.py

Then at the "mcp>" prompt:
    list                       - list available tools with their input schema
    call <tool_namelist
    > {"k":"v"} - call a tool with JSON arguments
    quit                       - exit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx2
from mcp.client.auth import AuthorizationCodeResult, OAuthClientProvider, TokenStorage
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

logging.basicConfig(level=logging.WARNING)

DEFAULT_SERVER_URL = "https://mcp.atlassian.com/v1/mcp/authv2"
DEFAULT_CALLBACK_PORT = 3030
DEFAULT_TOKEN_CACHE = Path(__file__).parent / ".atlassian_mcp_tokens.json"


class FileTokenStorage(TokenStorage):
    """Persists OAuth tokens and the DCR-registered client info to a JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._tokens: OAuthToken | None = None
        self._client_info: OAuthClientInformationFull | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        if data.get("tokens"):
            self._tokens = OAuthToken.model_validate(data["tokens"])
        if data.get("client_info"):
            self._client_info = OAuthClientInformationFull.model_validate(data["client_info"])

    def _save(self) -> None:
        data = {
            "tokens": self._tokens.model_dump(mode="json") if self._tokens else None,
            "client_info": self._client_info.model_dump(mode="json") if self._client_info else None,
        }
        self.path.write_text(json.dumps(data, indent=2))
        self.path.chmod(0o600)

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._tokens = tokens
        self._save()

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_info = client_info
        self._save()


class _CallbackHandler(BaseHTTPRequestHandler):
    """Captures the `code`/`state`/`error` query params from the OAuth redirect."""

    def __init__(self, *args: Any, callback_data: dict[str, Any], **kwargs: Any) -> None:
        self.callback_data = callback_data
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        if "code" in query:
            self.callback_data["code"] = query["code"][0]
            self.callback_data["state"] = query.get("state", [None])[0]
            self.callback_data["iss"] = query.get("iss", [None])[0]
            body = b"<html><body><h1>Authorized</h1><p>You can close this tab and return to the terminal.</p></body></html>"
            self.send_response(200)
        elif "error" in query:
            self.callback_data["error"] = query["error"][0]
            body = f"<html><body><h1>Authorization failed</h1><p>{query['error'][0]}</p></body></html>".encode()
            self.send_response(400)
        else:
            self.send_response(404)
            body = b""
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass  # silence default request logging


class CallbackServer:
    """Local HTTP server that catches the browser's OAuth redirect."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.callback_data: dict[str, Any] = {}
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def handler_factory(*args: Any, **kwargs: Any) -> _CallbackHandler:
            return _CallbackHandler(*args, callback_data=self.callback_data, **kwargs)

        self._server = HTTPServer(("localhost", self.port), handler_factory)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=1)

    def wait_for_code(self, timeout: float = 300) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if "code" in self.callback_data:
                return self.callback_data["code"]
            if "error" in self.callback_data:
                raise RuntimeError(f"OAuth authorization failed: {self.callback_data['error']}")
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for the OAuth browser redirect")


def build_oauth_provider(server_url: str, callback_port: int, token_cache: Path) -> OAuthClientProvider:
    callback_server = CallbackServer(callback_port)

    async def redirect_handler(authorization_url: str) -> None:
        print(f"\nOpening your browser to authorize this client:\n  {authorization_url}\n")
        webbrowser.open(authorization_url)

    async def callback_handler() -> AuthorizationCodeResult:
        callback_server.start()
        print(f"Waiting for the OAuth redirect on http://localhost:{callback_port}/callback ...")
        try:
            code = callback_server.wait_for_code()
        finally:
            callback_server.stop()
        return AuthorizationCodeResult(
            code=code,
            state=callback_server.callback_data.get("state"),
            iss=callback_server.callback_data.get("iss"),
        )

    client_metadata = OAuthClientMetadata.model_validate(
        {
            "client_name": "Atlassian MCP CLI Client",
            "redirect_uris": [f"http://localhost:{callback_port}/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        }
    )

    # Pass the actual MCP endpoint (not a guessed base URL) - the SDK
    # discovers protected-resource / authorization-server metadata from it,
    # with the well-known fallbacks defined by RFC 9728 / SEP-985.
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=FileTokenStorage(token_cache),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


def _print_tool_result(result: Any) -> None:
    content = getattr(result, "content", None)
    if not content:
        print(result)
        return
    for item in content:
        if getattr(item, "type", None) == "text":
            print(item.text)
        else:
            print(item)
    if getattr(result, "isError", False):
        print("(tool reported an error)")


async def interactive_loop(session: ClientSession) -> None:
    print("\nConnected. Commands:")
    print("  list                          - list available tools + input schema")
    print("  call <tool_name> {json args}  - call a tool")
    print("  quit                          - exit\n")

    while True:
        try:
            command = input("mcp> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not command:
            continue
        if command in ("quit", "exit"):
            break

        if command == "list":
            result = await session.list_tools()
            for i, tool in enumerate(result.tools, 1):
                print(f"{i}. {tool.name}")
                if tool.description:
                    print(f"   {tool.description}")
                if tool.input_schema:
                    print(f"   input schema: {json.dumps(tool.input_schema)}")
                print()
            continue

        if command.startswith("call "):
            parts = command.split(maxsplit=2)
            tool_name = parts[1] if len(parts) > 1 else ""
            if not tool_name:
                print("Please specify a tool name.")
                continue
            arguments: dict[str, Any] = {}
            if len(parts) > 2:
                try:
                    arguments = json.loads(parts[2])
                except json.JSONDecodeError as exc:
                    print(f"Invalid JSON arguments: {exc}")
                    continue
            try:
                result = await session.call_tool(tool_name, arguments)
            except Exception as exc:  # noqa: BLE001 - surface any tool-call failure to the user
                print(f"Tool call failed: {exc}")
                continue
            _print_tool_result(result)
            continue

        print("Unknown command. Try 'list', 'call <tool_name> {...}', or 'quit'.")


async def run(server_url: str, callback_port: int, token_cache: Path) -> None:
    oauth_auth = build_oauth_provider(server_url, callback_port, token_cache)

    print(f"Connecting to {server_url} ...")
    async with httpx2.AsyncClient(auth=oauth_auth, follow_redirects=True) as http_client:
        async with streamable_http_client(url=server_url, http_client=http_client) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                await interactive_loop(session)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Atlassian remote MCP server URL")
    parser.add_argument("--callback-port", type=int, default=DEFAULT_CALLBACK_PORT, help="Local port for the OAuth redirect")
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE, help="Where to persist OAuth tokens")
    args = parser.parse_args()

    try:
        asyncio.run(run(args.server_url, args.callback_port, args.token_cache))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
