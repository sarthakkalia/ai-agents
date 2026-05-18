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

RESULTS_DIR= "./results"

async def validator_agent(
    session: ClientSession, tools: list[dict], invoices: list[dict], agent_loop
) -> list[dict]:
    system = """You are a financial validation agent. For each invoice:
    1. Use load_all_results to load saved results for cross-checking
    2. Verify: total = subtotal + tax_amount
    3. Verify: each line_item.total ≈ quantity × unit_price
    4. Check for missing required fields (invoice_number, vendor_name, total_amount)
    5. Flag any [MISMATCH] tags in descriptions

    Return a JSON array of validation results:
    [{
    "invoice_number":    "string",
    "source_file":       "string",
    "status":            "valid" | "needs_review" | "invalid",
    "issues":            ["list of problem strings"],
    "confidence_score":  0.0 to 1.0
    }]
    Return ONLY the JSON array."""

    invoices_summary = json.dumps([
        {k: v for k, v in inv.items() if k != "line_items"}
        for inv in invoices
    ], indent=2)

    result = await agent_loop(
        session=session,
        agent_name="Validator",
        system_prompt=system,
        first_message=f"Validate these {len(invoices)} invoices:\n{invoices_summary}\n\nResults dir: {RESULTS_DIR}",
        tools=tools,
    )

    try:
        start = result.find("[")
        end   = result.rfind("]") + 1
        raw_list = json.loads(result[start:end])
        validated = [ValidationResult(**item).model_dump() for item in raw_list]
        return validated
    except Exception as e:
        Console().print(f"[red]Validation parsing failed: {e}[/red]")
        return []