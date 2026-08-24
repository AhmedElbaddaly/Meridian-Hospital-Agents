"""
run_platform.py -- launcher that sets the Windows Proactor event loop
policy BEFORE uvicorn creates its event loop. Required because
agent/agent.py spawns the MCP server via asyncio subprocess, which needs
Proactor; uvicorn's own startup picks Selector by default on Windows,
and setting the policy from inside main.py (imported after uvicorn's
loop already exists) is too late.

Run with: python run_platform.py
"""
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run("webapp.backend.main:app", host="127.0.0.1", port=8000, reload=False)