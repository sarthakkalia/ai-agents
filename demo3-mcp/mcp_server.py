import json
import sys
import os
from pathlib import Path
import pdfplumber
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import mcp.types as types
import asyncio

server = Server("invoice-analysis")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="list_invoice_files",
            description="List all invoice files in the 'invoices' directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Path to the folder containing invoice PDFs",
                    },
                },
                "required": ["directory"],
            }
        ),
        Tool(
            name="read_pdf_text",
            description="Extracts all text from a PDF file. Returns the raw text content page by page.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Full path to the PDF file",
                    }
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="save_result",
            description="Saves extracted invoice data as a JSON file in the results directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename":    {"type": "string", "description": "Output filename, e.g. 'invoice_001.json'"},
                    "data":        {"type": "string", "description": "JSON string of the extracted invoice data"},
                    "results_dir": {"type": "string", "description": "Directory to save results into"},
                },
                "required": ["filename", "data", "results_dir"],
            },
        ),
        Tool(
            name="load_all_results",
            description="Loads all previously saved JSON extraction results from the results directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "results_dir": {
                        "type": "string",
                        "description": "Directory containing JSON result files",
                    }
                },
                "required": ["results_dir"],
            },
        ),
        Tool(
            name="get_directory_stats",
            description="Returns metadata about a directory: number of files, total size, file list with sizes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Path to the directory to inspect",
                    }
                },
                "required": ["directory"],
            },
        )
    ]

@server.call_tool()
async def call_tool(name, args):
    if name == "list_invoice_files":
        directory = Path(args["directory"])

        if not directory.exists():
            return [TextContent(type="text", text=f"ERROR: Directory '{directory}' does not exist.")]
        # TextContent — the MCP standard response format

        pdf_files = sorted(directory.glob("*.pdf"))

        if not pdf_files:
            return [TextContent(type="text", text=f"No PDF files found in '{directory}'.")]
 
        files_info = []
        for f in pdf_files:
            stat = f.stat()
            files_info.append({
                "filename":  f.name,
                "full_path": str(f.resolve()),
                "size_kb":   round(stat.st_size / 1024, 2),
            })
            
        result = {
            "directory":  str(directory.resolve()),
            "file_count": len(files_info),
            "files":      files_info,
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "read_pdf_text":
        file_path = Path(args["file_path"])

        if not file_path.exists():
            return [TextContent(type="text", text=f"ERROR: File '{file_path}' not found.")]
        try:
            pages_text = []
            with pdfplumber.open(str(file_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    table_text = ""
                    for table in tables:
                        for row in table:
                            if row:
                                table_text += " | ".join(str(c) for c in row if c) + "\n"

                    pages_text.append({
                        "page":       page_num,
                        "text":       text.strip(),
                        "table_data": table_text.strip(),
                    })
            result = {
                "file":       file_path.name,
                "page_count": len(pages_text),
                "pages":      pages_text,
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        except Exception as e:
            return [TextContent(type="text", text=f"ERROR reading PDF: {e}")]

    elif name == "load_all_results":
        results_dir = Path(args["results_dir"])
        if not results_dir.exists():
            return [TextContent(type="text", text=f"ERROR: Results directory '{results_dir}' does not exist.")]
        
        json_files = sorted(results_dir.glob("*.json"))
        if not json_files:
            return [TextContent(type="text", text="No result files found.")]
        
        all_results = []
        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
                    all_results.append({
                        "filename": jf.name,
                        "data": data,
                    })
            except Exception as e:
                return [TextContent(type="text", text=f"ERROR reading result file '{jf}': {e}")]
        return [TextContent(type="text", text=json.dumps(all_results, indent=2))]
    
    elif name == "save_result":
        results_dir = Path(args["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
 
        output_path = results_dir / args["filename"]
        try:
            # Validate that 'data' is valid JSON before saving
            parsed = json.loads(args["data"])
            with open(output_path, "w") as f:
                json.dump(parsed, f, indent=2)
            return [TextContent(type="text", text=f"Saved to: {output_path}")]
        except json.JSONDecodeError as e:
            return [TextContent(type="text", text=f"ERROR: data is not valid JSON — {e}")]
        except Exception as e:
            return [TextContent(type="text", text=f"ERROR saving file: {e}")]

    elif name == "get_directory_stats":
        directory = Path(args["directory"])
        if not directory.exists():
            return [TextContent(type="text", text=f"ERROR: Directory '{directory}' does not exist.")]
        all_files = list(directory.iterdir())
        total_size = sum(f.stat().st_size for f in all_files if f.is_file())

        stats= {
            "directory":       str(directory.resolve()),
            "total_files":     len(all_files),
            "total_size_kb":   round(total_size / 1024, 2),
            "files": [
                {"name": f.name, "size_kb": round(f.stat().st_size / 1024, 2)}
                for f in sorted(all_files) if f.is_file()
            ],
        }
        return [TextContent(type="text", text=json.dumps(stats, indent=2))]
    
    else:
        return [TextContent(type="text", text=f"ERROR: Unknown tool '{name}'")]
    
async def main():
    async with stdio_server() as (read_stream, write_stream):
        # stdio_server sets up the MCP server to communicate 
        # over standard input/output,                          ## sys.stdin → MCP protocol reader
        # which is ideal for local testing and development.    ## MCP protocol writer → sys.stdout
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main())