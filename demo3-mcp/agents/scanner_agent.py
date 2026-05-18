import asyncio
import json
from aiohttp import ClientSession
from groq import Groq
from mcp_server import Tool
from orchestrator import agent_loop
from rich.console import Console


INVOICES_DIR = "./invoices"

async def scanner_agent(session: ClientSession, tools: list[dict], agent_loop  ):
    system = """You are a file scanner agent. Your job is to:
    1. Use the list_invoice_files tool to discover all PDF files in the given directory
    2. Use get_directory_stats for a summary
    3. Return ONLY a JSON array of file objects — no other text.
    Format: [{"filename": "...", "full_path": "...", "size_kb": 0.0}, ...]"""

    result = await agent_loop(
        session=session,
        agent_name="Scanner",
        system_prompt=system,
        first_message=f"Scan the invoices directory at: {INVOICES_DIR}",
        tools=tools,
    )

    try:
        start = result.find("[")
        end   = result.rfind("]") + 1
        return json.loads(result[start:end])
    except Exception:
        Console().print(f"[red]Scanner returned non-JSON: {result[:200]}[/red]")
        return []
