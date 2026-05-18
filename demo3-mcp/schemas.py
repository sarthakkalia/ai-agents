from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

class LineItem(BaseModel):
    description: str = Field(description="Name or description of the product/service")
    quantity: float = Field(description="Number of units", gt=0)
    unit_price: float = Field(description="Price per unit", ge=0)
    total: float = Field(description="quantity × unit_price")

    @field_validator("total", mode="before")
    @classmethod
    def coerce_total(cls, v) -> float:
        if isinstance(v, str):
            return float(v.replace(",", "").replace("$", "").strip())
        return float(v)
 
    @model_validator(mode="after")
    def check_line_total(self) -> "LineItem":
        expected = round(self.quantity * self.unit_price, 2)
        actual   = round(self.total, 2)
        if abs(expected - actual) > max(0.01, expected * 0.01):
            self.description = f"[MISMATCH] {self.description}"
        return self
    
class InvoiceData(BaseModel):
    invoice_number: str = Field(description="Invoice ID, e.g. 'INV-2024-001'")
    source_file: str    = Field(description="Original PDF filename")

    vendor_name:  str           = Field(description="Company that issued the invoice")
    vendor_email: Optional[str] = Field(default=None, description="Vendor contact email")

    invoice_date: str           = Field(description="Date invoice was issued, YYYY-MM-DD format")
    due_date:     Optional[str] = Field(default=None, description="Payment due date, YYYY-MM-DD")

    line_items: list[LineItem] = Field(
        description="All products/services listed on the invoice",
        min_length=1,
    )

    subtotal:     float          = Field(description="Sum before tax", ge=0)
    tax_amount:   Optional[float] = Field(default=None, description="Tax charged", ge=0)
    total_amount: float          = Field(description="Final amount due", ge=0)
    currency:     str            = Field(default="USD", description="Currency code e.g. USD, EUR")

    extraction_notes: Optional[str] = Field(
        default=None,
        description="Any caveats about this extraction (e.g. blurry scan, missing fields)",
    )

    @field_validator("invoice_date", "due_date", mode="before")
    @classmethod
    def normalise_date(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        import re
        if re.match(r"\d{4}-\d{2}-\d{2}", v):
            return v
        # Try MM/DD/YYYY or DD/MM/YYYY
        match = re.match(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", v)
        if match:
            m, d, y = match.groups()
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        return v
    
    @model_validator(mode="after")
    def validate_total_math(self) -> "InvoiceData":
        tax = self.tax_amount or 0.0
        expected_total = round(self.subtotal + tax, 2)
        if abs(expected_total - self.total_amount) > 0.05:
            note = f"Total mismatch: subtotal({self.subtotal}) + tax({tax}) = {expected_total} but total_amount={self.total_amount}"
            self.extraction_notes = (self.extraction_notes or "") + " | " + note
        return self
    
    def to_dict(self):
        return self.model_dump()
    

class ValidationResult(BaseModel):
    invoice_number: str
    source_file:    str
    status: Literal["valid", "needs_review", "invalid"] = Field(
        description="valid = all checks pass | needs_review = minor issues | invalid = major errors"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of specific problems found (empty if status=valid)",
    )
    confidence_score: float = Field(
        description="0.0–1.0 — how confident the agent is in the extracted data",
        ge=0.0, le=1.0,
    )

class InvoiceSummary(BaseModel):
    total_invoices:   int
    valid_invoices:   int
    flagged_invoices: int
    invalid_invoices: int
 
    total_spend:     float = Field(description="Sum of all invoice totals")
    average_invoice: float = Field(description="Mean invoice value")
    currency:        str   = Field(default="USD")
 
    top_vendors: list[dict] = Field(
        description="[{vendor_name, total_billed, invoice_count}] sorted by spend desc",
    )
    date_range: dict = Field(
        description="{earliest: YYYY-MM-DD, latest: YYYY-MM-DD}",
    )
    key_insights: list[str] = Field(
        description="3–5 bullet-point insights the agent noticed",
        min_length=1,
    )