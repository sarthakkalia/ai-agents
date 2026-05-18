import asyncio
import json
from pathlib import Path
from mcp import ClientSession
from groq import Groq
import json
from mcp_server import Tool
from rich.console import Console
from pydantic import BaseModel
from schemas import InvoiceData

RESULTS_DIR = "./results"

async def extractor_agent(
    session: ClientSession, tools: list[dict], file_path: Path, agent_loop  
):
    system = f"""You are a medical invoice extraction agent. Your job is to:
    1. You MUST call the read_pdf_text tool before answering.Do not guess invoice data.
    If you do not call the tool, your answer is invalid.
    Return ONLY valid JSON.
    2. Extract ALL invoice fields from the text
    3. Return a JSON object matching this EXACT schema (no extra fields):
    {{
    "invoice_number":   "string",
    "source_file":      "string (filename only)",
    "vendor_name":      "string",
    "vendor_email":     "string or null",
    "invoice_date":     "YYYY-MM-DD",
    "due_date":         "YYYY-MM-DD or null",
    "line_items": [
        {{"description": "str", "quantity": 1.0, "unit_price": 0.0, "total": 0.0}}
    ],
    "subtotal":         0.0,
    "tax_amount":       0.0,
    "total_amount":     0.0,
    "currency":         "USD",
    "extraction_notes": "any caveats or null"
    }}
    Return ONLY the JSON object — no markdown, no explanation."""

    result = await agent_loop(
        session=session,
        agent_name=f"Extractor ({file_path.name})",
        system_prompt=system,
        first_message=f"Extract invoice data from this file: {file_path}",
        tools=tools,
    )

    try:
        start = result.find("{")
        end   = result.rfind("}") + 1
        raw   = json.loads(result[start:end])

        invoice = InvoiceData(**raw)
        data    = invoice.to_dict()

        # Save via MCP tool
        save_msg = await session.call_tool("save_result", {
            "filename":    f"{file_path.stem}.json",
            "data":        json.dumps(data),
            "results_dir": RESULTS_DIR,
        })
        Console().print(f"  [green]Saved:[/green] {save_msg.content[0].text}")
        return data

    except Exception as e:
        Console().print(f"  [red]Extraction failed for {file_path.name}: {e}[/red]")
        return None