import asyncio
import json
from pathlib import Path
from mcp import ClientSession
from aiohttp import ClientSession
from groq import Groq
from mcp_server import Tool
from rich.console import Console
from pydantic import BaseModel
from schemas import InvoiceData, ValidationResult, InvoiceSummary

RESULTS_DIR = "./results"

async def summarizer_agent(
    session: ClientSession, tools: list[dict],
    invoices: list[dict], validations: list[dict], agent_loop  
) -> dict:
    """
    Combines all invoice data and validation results into a single
    InvoiceSummary report, then saves it.
    """
    system = """You are a financial summarizer agent. Analyze all invoices and return:
{
  "total_invoices":   number,
  "valid_invoices":   number,
  "flagged_invoices": number,
  "invalid_invoices": number,
  "total_spend":      number,
  "average_invoice":  number,
  "currency":         "USD",
  "top_vendors": [
    {"vendor_name": "str", "total_billed": 0.0, "invoice_count": 0}
  ],
  "date_range": {"earliest": "YYYY-MM-DD", "latest": "YYYY-MM-DD"},
  "key_insights": ["insight 1", "insight 2", "insight 3"]
}
Return ONLY the JSON object."""

    combined = {
        "invoices":    invoices,
        "validations": validations,
    }

    result = await agent_loop(
        session=session,
        agent_name="Summarizer",
        system_prompt=system,
        first_message=f"Summarize this invoice batch:\n{json.dumps(combined, indent=2)[:6000]}",
        tools=tools,
    )

    try:
        start = result.find("{")
        end   = result.rfind("}") + 1
        raw   = json.loads(result[start:end])
        summary = InvoiceSummary(**raw)
        data    = summary.model_dump()

        # Save the summary report
        await session.call_tool("save_result", {
            "filename":    "_summary_report.json",
            "data":        json.dumps(data),
            "results_dir": RESULTS_DIR,
        })
        return data
    except Exception as e:
        Console().print(f"[red]Summary failed: {e}[/red]")
        return {}

