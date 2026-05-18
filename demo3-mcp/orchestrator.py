import asyncio
import json
import os
import sys
from pathlib import Path
from time import time
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
 
from schemas import InvoiceData, ValidationResult, InvoiceSummary

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from groq import Groq
import groq as groq_lib

import os
from cerebras.cloud.sdk import Cerebras

import asyncio
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.extractor_agent import extractor_agent
from agents.validator_agent import validator_agent
from agents.summarizer_agent import summarizer_agent


INVOICES_DIR      = "./invoices"
RESULTS_DIR       = "./results"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# MODEL        = "llama-3.3-70b-versatile"
MODEL= "llama3.1-8b"

console = Console()
# client = Groq(api_key=GROQ_API_KEY)
client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)

def mcp_to_groq_tool(mcp_tool):
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
        },
    }

async def agent_loop(
    session: ClientSession, agent_name: str, system_prompt: str, 
    first_message: str, tools: list[dict], max_iterations: int = 3,):

    console.print(f"\n[bold blue]► Agent: {agent_name}[/bold blue]")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": first_message},
    ]

    for iteration in range(max_iterations):
        console.print(f"  [dim]iteration {iteration + 1}...[/dim]")
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            tools=tools,
            tool_choice="required",
            messages=messages,
        )

        choice      = response.choices[0]
        finish      = choice.finish_reason
        msg         = choice.message

        assistant_msg = {"role": "assistant", "content": msg.content or ""}

        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
            {
                "id":   tc.id,
                "type": "function",
                "function": {
                    "name":      tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
        messages.append(assistant_msg)

        if finish == "stop":
            if msg.content and msg.content.strip():
                console.print(f"  [green]✓ Done ({len(messages)} messages)[/green]")
                return msg.content
            console.print("  [yellow]Stop received but no content yet...[/yellow]")
            continue
        
        if finish == "tool_calls":
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                console.print(f"  [yellow]→ Calling tool:[/yellow] {tool_name}({json.dumps(tool_args)[:80]})")

                try:
                    mcp_result  = await session.call_tool(tool_name, tool_args)
                    result_text = mcp_result.content[0].text if mcp_result.content else "No result"
                    console.print(f"  [dim]← {result_text[:120]}[/dim]")
                except Exception as e:
                    result_text = f"Tool error: {e}"
                    console.print(f"  [red]✗ {e}[/red]")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "content": result_text,
                })

        else:
            console.print(f"  [red]Unexpected finish_reason: {finish}[/red]")
            break
 
    return "Agent reached max iterations without completing."


def print_validation_table(validations: list[dict]):
    table = Table(title="Validation Results", show_header=True)
    table.add_column("Invoice #",   style="cyan")
    table.add_column("File",        style="dim")
    table.add_column("Status",      style="bold")
    table.add_column("Confidence",  justify="right")
    table.add_column("Issues")

    for v in validations:
        status_color = {"valid": "green", "needs_review": "yellow", "invalid": "red"}.get(v["status"], "white")
        table.add_row(
            v.get("invoice_number", "?"),
            v.get("source_file", "?"),
            f"[{status_color}]{v['status']}[/{status_color}]",
            f"{v.get('confidence_score', 0):.0%}",
            "; ".join(v.get("issues", [])) or "—",
        )
    console.print(table)

def print_summary(summary: dict):
    console.print(Panel.fit(
        f"[bold]Total invoices:[/bold]   {summary.get('total_invoices', 0)}\n"
        f"[green]Valid:[/green]              {summary.get('valid_invoices', 0)}\n"
        f"[yellow]Needs review:[/yellow]     {summary.get('flagged_invoices', 0)}\n"
        f"[red]Invalid:[/red]           {summary.get('invalid_invoices', 0)}\n\n"
        f"[bold]Total spend:[/bold]       ${summary.get('total_spend', 0):,.2f} {summary.get('currency','')}\n"
        f"[bold]Average invoice:[/bold]  ${summary.get('average_invoice', 0):,.2f}\n\n"
        f"[bold]Date range:[/bold]       {summary.get('date_range', {}).get('earliest','?')} → {summary.get('date_range', {}).get('latest','?')}\n\n"
        f"[bold]Key insights:[/bold]\n" + "\n".join(f"  • {i}" for i in summary.get("key_insights", [])),
        title="[bold blue]Invoice Batch Summary[/bold blue]",
        border_style="blue",
    ))

    if summary.get("top_vendors"):
        vtable = Table(title="Top Vendors by Spend")
        vtable.add_column("Vendor",    style="cyan")
        vtable.add_column("Invoices",  justify="right")
        vtable.add_column("Total",     justify="right", style="green")
        for v in summary["top_vendors"]:
            vtable.add_row(v["vendor_name"], str(v["invoice_count"]), f"${v['total_billed']:,.2f}")
        console.print(vtable)


async def main():
    console.print(Panel.fit("[bold blue]Multi-Agent Invoice Extractor[/bold blue]\nUsing MCP + GROQ", border_style="blue"))

    if not GROQ_API_KEY:
        console.print("[red]GROQ_API_KEY not set. Check your .env file.[/red]")
        sys.exit(1)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=None,
    )
    
    console.print("\n[dim]Starting MCP server subprocess...[/dim]")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            console.print("[green]✓ MCP server connected[/green]")

            tools_response = await session.list_tools()
            mcp_tools      = tools_response.tools
            anthropic_tools = [mcp_to_groq_tool(t) for t in mcp_tools]
            console.print(f"[green]✓ {len(anthropic_tools)} tools available: {[t['function']['name'] for t in anthropic_tools]}[/green]")

            #  agent-1: Scanner
            console.rule("[bold]Phase 1: Scanning[/bold]")
            files = sorted(Path(INVOICES_DIR).glob("*.pdf"))
            console.print(f"Found [bold]{len(files)}[/bold] invoice files in '{INVOICES_DIR}'")

            if not files:
                console.print("[red]No invoice files found. Run:  python create_samples.py[/red]")
                return
            
            console.print(f"\n[green]Found {len(files)} invoice(s):[/green]")
            for f in files:
                console.print(f"  • {f.name} ({f.stat().st_size // 1024} KB)")
            
            # agent-2: Extractor
            console.rule("[bold]Phase 2: Extracting[/bold]")
            invoices = []
            for file_info in files:
                data = await extractor_agent(session, anthropic_tools, file_info, agent_loop)
                if data:
                    invoices.append(data)
                    console.print(f"  [green]✓[/green] Extracted: {data.get('invoice_number')} — ${data.get('total_amount', 0):,.2f}")

            if not invoices:
                console.print("[red]No invoices successfully extracted.[/red]")
                return
            
            # agent-3: Validator
            console.rule("[bold]Phase 3: Validating[/bold]")
            validations = await validator_agent(session, anthropic_tools, invoices, agent_loop)
            print_validation_table(validations)

            # agent-4: Summarizer
            console.rule("[bold]Phase 4: Summarizing[/bold]")
            summary = await summarizer_agent(session, anthropic_tools, invoices, validations, agent_loop)
            print_summary(summary)

            console.print(f"\n[bold green]Complete![/bold green] Results saved to: [cyan]{RESULTS_DIR}/[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())