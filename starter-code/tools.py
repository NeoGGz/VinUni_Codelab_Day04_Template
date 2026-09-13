import json
import os
from typing import List, Dict, Any
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# Tool #1: đọc product_catalog.json và lọc theo category, max_price.
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.isfile(catalog_file):
        return []

    with open(catalog_file, "r", encoding="utf-8") as file:
        products = json.load(file)

    if not isinstance(products, list):
        return []

    return [
        product
        for product in products
        if product.get("category") == category
        and isinstance(product.get("price_vnd"), (int, float))
        and product["price_vnd"] <= max_price
    ]


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# Tool #2: tạo ticket mới và lưu vào support_tickets.json.
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")
    priority = priority.lower().strip()
    if priority not in {"low", "medium", "high"}:
        priority = "medium"

    tickets = []
    if os.path.isfile(tickets_file):
        with open(tickets_file, "r", encoding="utf-8") as file:
            loaded_tickets = json.load(file)
            if isinstance(loaded_tickets, list):
                tickets = loaded_tickets

    sequence_numbers = []
    for existing_ticket in tickets:
        ticket_id = str(existing_ticket.get("ticket_id", ""))
        try:
            sequence_numbers.append(int(ticket_id.rsplit("-", 1)[-1]))
        except ValueError:
            continue

    now = datetime.now().astimezone()
    sequence = max(sequence_numbers, default=0) + 1
    ticket = {
        "ticket_id": f"TK-{now.strftime('%Y%m%d')}-{sequence:03d}",
        "customer_name": customer_name.strip(),
        "issue_description": issue_description.strip(),
        "priority": priority,
        "status": "open",
        "created_at": now.isoformat(),
        "category": "general",
    }

    tickets.append(ticket)
    with open(tickets_file, "w", encoding="utf-8") as file:
        json.dump(tickets, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return ticket


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# JSON Schema cho từng tool (name, description, parameters).
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": "Tra cứu sản phẩm hoặc dịch vụ Vingroup theo danh mục và giá tối đa.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["xe_dien", "du_lich"],
                    "description": "Loại sản phẩm: xe điện hoặc gói du lịch Vinpearl.",
                },
                "max_price": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Giá tối đa tính bằng VNĐ.",
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_support_ticket",
        "description": "Ghi nhận và tạo yêu cầu hỗ trợ cho khách hàng.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Tên khách hàng.",
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả vấn đề cần hỗ trợ.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                    "description": "Mức độ ưu tiên.",
                },
            },
            "required": ["customer_name", "issue_description"],
        },
    },
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}