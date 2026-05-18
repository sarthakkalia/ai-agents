"""
python create_samples.py
Creates 4 realistic sample invoice PDFs in ./invoices/
  - invoice_techcorp.pdf      (software services, multi-line)
  - invoice_cloudhost.pdf     (hosting + support, with tax)
  - invoice_designstudio.pdf  (design work, USD)
  - invoice_broken.pdf        (intentionally has a math error — for Validator to catch)
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import cm
from pathlib import Path

OUTPUT_DIR = Path("./invoices")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_invoice_pdf(filename: str, data: dict):
    doc = SimpleDocTemplate(str(OUTPUT_DIR / filename), pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    header_style  = ParagraphStyle("Header",  fontSize=22, fontName="Helvetica-Bold",  textColor=colors.HexColor("#1a1a2e"))
    vendor_style  = ParagraphStyle("Vendor",  fontSize=11, fontName="Helvetica",       textColor=colors.HexColor("#444"))
    label_style   = ParagraphStyle("Label",   fontSize=10, fontName="Helvetica-Bold",  textColor=colors.HexColor("#333"))
    value_style   = ParagraphStyle("Value",   fontSize=10, fontName="Helvetica",       textColor=colors.HexColor("#555"))

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph(data["vendor_name"], header_style))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(data.get("vendor_address", ""), vendor_style))
    story.append(Paragraph(data.get("vendor_email", ""), vendor_style))
    story.append(Spacer(1, 1*cm))

    # ── Invoice meta ──────────────────────────────────────────────────────────
    meta_data = [
        ["Invoice Number:", data["invoice_number"],  "Invoice Date:", data["invoice_date"]],
        ["Bill To:",        data["bill_to"],          "Due Date:",     data["due_date"]],
    ]
    meta_table = Table(meta_data, colWidths=[3.5*cm, 6*cm, 3*cm, 4*cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",   (0, 0), (-1, -1), colors.HexColor("#333")),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 1*cm))

    # ── Line items table ──────────────────────────────────────────────────────
    header_row = ["Description", "Qty", "Unit Price", "Total"]
    rows = [header_row]
    for item in data["line_items"]:
        rows.append([
            item["description"],
            str(item["qty"]),
            f"${item['unit_price']:.2f}",
            f"${item['total']:.2f}",
        ])

    items_table = Table(rows, colWidths=[9*cm, 2*cm, 3.5*cm, 3*cm])
    items_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("ALIGN",        (1, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f8f8"), colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#ddd")),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    story.append(items_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Totals ────────────────────────────────────────────────────────────────
    totals = [
        ["", "Subtotal:", f"${data['subtotal']:.2f}"],
        ["", f"Tax ({data.get('tax_pct', 0)}%):", f"${data.get('tax_amount', 0):.2f}"],
        ["", "TOTAL DUE:", f"${data['total']:.2f}"],
    ]
    totals_table = Table(totals, colWidths=[9.5*cm, 4*cm, 4*cm])
    totals_table.setStyle(TableStyle([
        ("FONTNAME",  (1, 2), (-1, 2), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 10),
        ("ALIGN",     (1, 0), (-1, -1), "RIGHT"),
        ("LINEABOVE", (1, 2), (-1, 2), 1, colors.HexColor("#1a1a2e")),
        ("TOPPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(totals_table)
    story.append(Spacer(1, 1.5*cm))

    # ── Notes ─────────────────────────────────────────────────────────────────
    if data.get("notes"):
        story.append(Paragraph("Notes:", label_style))
        story.append(Paragraph(data["notes"], value_style))

    doc.build(story)
    print(f"  Created: {OUTPUT_DIR / filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Sample invoice data
# ─────────────────────────────────────────────────────────────────────────────
INVOICES = [
    {
        "filename": "invoice_techcorp.pdf",
        "vendor_name":    "TechCorp Solutions Ltd.",
        "vendor_address": "500 Innovation Drive, San Francisco, CA 94105",
        "vendor_email":   "billing@techcorp.io",
        "invoice_number": "INV-2024-001",
        "invoice_date":   "2024-11-01",
        "due_date":       "2024-11-30",
        "bill_to":        "Acme Inc.",
        "line_items": [
            {"description": "Software Development (Backend API)", "qty": 40, "unit_price": 150.00, "total": 6000.00},
            {"description": "DevOps & Infrastructure Setup",       "qty": 8,  "unit_price": 175.00, "total": 1400.00},
            {"description": "Code Review & Documentation",         "qty": 4,  "unit_price": 100.00, "total": 400.00},
        ],
        "subtotal":    7800.00,
        "tax_pct":     0,
        "tax_amount":  0.00,
        "total":       7800.00,
        "notes":       "Payment via bank transfer. PO# 2024-TC-88.",
    },
    {
        "filename": "invoice_cloudhost.pdf",
        "vendor_name":    "CloudHost Pro",
        "vendor_address": "1200 Data Center Blvd, Austin, TX 78701",
        "vendor_email":   "accounts@cloudhost.pro",
        "invoice_number": "INV-CH-2024-Q4",
        "invoice_date":   "2024-10-15",
        "due_date":       "2024-11-15",
        "bill_to":        "Acme Inc.",
        "line_items": [
            {"description": "Cloud Server (4 vCPU, 16GB RAM) — Oct",  "qty": 1, "unit_price": 480.00, "total": 480.00},
            {"description": "Managed Database Service — Oct",           "qty": 1, "unit_price": 220.00, "total": 220.00},
            {"description": "CDN Bandwidth (500GB)",                    "qty": 1, "unit_price": 45.00,  "total": 45.00},
            {"description": "24/7 Support Plan — Oct",                  "qty": 1, "unit_price": 99.00,  "total": 99.00},
        ],
        "subtotal":    844.00,
        "tax_pct":     8,
        "tax_amount":  67.52,
        "total":       911.52,
        "notes":       "Auto-renewed monthly subscription. Cancel 30 days notice.",
    },
    {
        "filename": "invoice_designstudio.pdf",
        "vendor_name":    "Pixel Perfect Design Studio",
        "vendor_address": "77 Creative Alley, New York, NY 10001",
        "vendor_email":   "hello@pixelperfect.design",
        "invoice_number": "PP-INV-0892",
        "invoice_date":   "2024-09-20",
        "due_date":       "2024-10-05",
        "bill_to":        "Acme Inc.",
        "line_items": [
            {"description": "Brand Identity Redesign",        "qty": 1, "unit_price": 2500.00, "total": 2500.00},
            {"description": "UI/UX Design (Mobile App)",      "qty": 1, "unit_price": 3200.00, "total": 3200.00},
            {"description": "Icon Set (48 custom icons)",     "qty": 48, "unit_price": 25.00,  "total": 1200.00},
            {"description": "Revision Rounds (3 included)",   "qty": 3,  "unit_price": 0.00,   "total": 0.00},
        ],
        "subtotal":    6900.00,
        "tax_pct":     0,
        "tax_amount":  0.00,
        "total":       6900.00,
        "notes":       "All design files delivered in Figma + exported assets.",
    },
    {
        "filename": "invoice_broken.pdf",
        "vendor_name":    "QuickFix Consulting",
        "vendor_address": "999 Error Lane, Bugsville, CA 90210",
        "vendor_email":   "wrong@quickfix.net",
        "invoice_number": "QF-ERR-001",
        "invoice_date":   "2024-10-01",
        "due_date":       "2024-10-31",
        "bill_to":        "Acme Inc.",
        "line_items": [
            {"description": "Consulting Hours",    "qty": 10, "unit_price": 200.00, "total": 2000.00},
            {"description": "Expenses",            "qty": 1,  "unit_price": 350.00, "total": 500.00},   # WRONG: 350 ≠ 500
        ],
        "subtotal":    2500.00,     # correct
        "tax_pct":     10,
        "tax_amount":  250.00,      # correct (10% of 2500)
        "total":       3000.00,     # WRONG: should be 2750
        "notes":       "This invoice has intentional math errors for the Validator to catch.",
    },
]


if __name__ == "__main__":
    print("Creating sample invoice PDFs...")
    for inv in INVOICES:
        build_invoice_pdf(inv.pop("filename"), inv)
    print(f"\nDone! {len(INVOICES)} PDFs created in ./invoices/")
    print("Now run:  python orchestrator.py")