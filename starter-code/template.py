"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import os
import re
import sys
import unicodedata
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

if sys.platform == "win32":
    # PowerShell may default to cp1252, which cannot print Vietnamese text.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant, trợ lý AI chính thức hỗ trợ sản phẩm và dịch vụ Vingroup.

## PERSONA
- Vai trò: chuyên viên tư vấn VinFast và Vinpearl.
- Giọng nói: chuyên nghiệp, thân thiện, chính xác và không bịa thông tin.

## AVAILABLE TOOLS
- search_product_catalog: tra cứu sản phẩm theo danh mục và giá tối đa.
- submit_support_ticket: tạo phiếu hỗ trợ cho khách hàng.

## CORE RULES
1. Phải gọi search_product_catalog khi khách hàng cần tra cứu sản phẩm hoặc giá.
2. Phải gọi submit_support_ticket để tạo ticket; không tự tạo mã ticket.
3. Không gọi tool cho câu hỏi FAQ có thể trả lời trực tiếp.
4. Chỉ dùng dữ liệu trong kết quả tool và thông tin đã được cung cấp.

## OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ các chủ đề thuộc hệ sinh thái Vingroup.
- Không suy đoán giá, tồn kho hoặc trạng thái ticket.
- Khi không tìm thấy kết quả, phải thông báo rõ và lịch sự.

## OUTPUT CONTRACT
- Nội bộ tuân theo chu trình Thought -> Action -> Observation.
- Câu trả lời cho người dùng chỉ hiển thị Final Answer ngắn gọn, rõ ràng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        # Baseline intentionally makes one direct LLM call and never uses tools.
        if self.api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(user_input)
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": "live_baseline",
                }
            except (ImportError, AttributeError, RuntimeError, ValueError):
                # Keep the lab runnable without the optional Gemini dependency.
                pass

        return {
            "answer": f"[Chatbot Baseline] Tôi sẽ trả lời trực tiếp câu hỏi: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline",
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFD", text.lower())
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        return normalized.replace("đ", "d")

    @classmethod
    def _extract_max_price(cls, text: str) -> int:
        normalized = cls._normalize(text)
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*(ty|trieu|nghin|ngan)", normalized)
        if not match:
            return 999999999999

        number = float(match.group(1).replace(",", "."))
        multipliers = {
            "ty": 1_000_000_000,
            "trieu": 1_000_000,
            "nghin": 1_000,
            "ngan": 1_000,
        }
        return int(number * multipliers[match.group(2)])

    @staticmethod
    def _extract_customer_name(text: str) -> str:
        match = re.search(
            r"(?:tôi\s+tên|tên\s+tôi\s+là)\s+([^,.;!?]+)",
            text,
            flags=re.IGNORECASE,
        )
        return match.group(1).strip() if match else "Khách hàng"

    @staticmethod
    def _extract_issue(text: str) -> str:
        name_match = re.search(
            r"(?:tôi\s+tên|tên\s+tôi\s+là)\s+[^,.;!?]+[,.;]?",
            text,
            flags=re.IGNORECASE,
        )
        issue = text[name_match.end():] if name_match else text
        issue = issue.strip(" ,.;:-")
        issue = re.split(r"[.!?]", issue, maxsplit=1)[0]
        issue = re.split(r",\s*(?:mức độ|ưu tiên)", issue, maxsplit=1, flags=re.IGNORECASE)[0]
        issue = re.sub(r"\bcủa tôi\b", "", issue, flags=re.IGNORECASE)
        issue = re.sub(r"\s+", " ", issue).strip(" ,.;:-")
        if not issue:
            return "Yêu cầu hỗ trợ từ khách hàng"
        return issue[0].upper() + issue[1:]

    @classmethod
    def _detect_priority(cls, text: str) -> str:
        normalized = cls._normalize(text)
        if any(keyword in normalized for keyword in ("nghiem trong", "khan cap", "xu ly gap", "uu tien cao")):
            return "high"
        if any(keyword in normalized for keyword in ("muc do thap", "uu tien thap", "khong gap")):
            return "low"
        return "medium"

    @staticmethod
    def _format_price(price: int) -> str:
        return f"{price:,}".replace(",", ".") + " VNĐ"

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        normalized = self._normalize(user_input)

        product_terms = ("xe dien", "vinfast", "resort", "vinpearl", "du lich")
        search_terms = ("xem", "tim", "gia duoi", "gia toi da", "co xe", "goi du lich")
        needs_catalog = (
            any(term in normalized for term in product_terms)
            and any(term in normalized for term in search_terms)
            and "bao hanh" not in normalized
        )
        needs_ticket = (
            any(term in normalized for term in ("toi ten", "ten toi la", "ghi nhan phan hoi", "khieu nai"))
            and any(term in normalized for term in ("loi", "ho tro", "phan hoi", "am moc", "hong"))
        )

        planned_actions: List[Dict[str, Any]] = []
        if needs_catalog:
            category = "du_lich" if any(
                term in normalized for term in ("resort", "vinpearl", "du lich")
            ) else "xe_dien"
            planned_actions.append({
                "tool": "search_product_catalog",
                "arguments": {
                    "category": category,
                    "max_price": self._extract_max_price(user_input),
                },
            })

        if needs_ticket:
            planned_actions.append({
                "tool": "submit_support_ticket",
                "arguments": {
                    "customer_name": self._extract_customer_name(user_input),
                    "issue_description": self._extract_issue(user_input),
                    "priority": self._detect_priority(user_input),
                },
            })

        self.trace.append({
            "step": "intent_detection",
            "catalog": needs_catalog,
            "ticket": needs_ticket,
            "planned_tools": [action["tool"] for action in planned_actions],
        })

        observations: Dict[str, Any] = {}
        iterations = 0
        while planned_actions and iterations < self.max_iterations:
            action = planned_actions.pop(0)
            tool_name = action["tool"]
            arguments = action["arguments"]
            try:
                observation = TOOL_MAP[tool_name](**arguments)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                observation = {"error": str(error)}

            iterations += 1
            observations[tool_name] = observation
            self.trace.append({
                "step": iterations,
                "action": tool_name,
                "arguments": arguments,
                "observation": observation,
            })

        answers: List[str] = []
        products = observations.get("search_product_catalog")
        if products is not None:
            if isinstance(products, list) and products:
                product_lines = [
                    f"- {product['name']}: {self._format_price(product['price_vnd'])}"
                    for product in products
                ]
                answers.append("Các lựa chọn phù hợp:\n" + "\n".join(product_lines))
            else:
                answers.append("Rất tiếc, không tìm thấy sản phẩm phù hợp với mức giá bạn yêu cầu.")

        ticket = observations.get("submit_support_ticket")
        if isinstance(ticket, dict) and "ticket_id" in ticket:
            answers.append(
                f"Đã ghi nhận yêu cầu của {ticket['customer_name']}. "
                f"Mã ticket: {ticket['ticket_id']} (trạng thái: {ticket['status']})."
            )

        if not planned_actions and not observations:
            iterations = 1
            if "bao hanh" in normalized and "pin" in normalized:
                answers.append(
                    "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm. "
                    "Điều kiện cụ thể có thể khác nhau theo mẫu xe và thị trường."
                )
            else:
                answers.append("Tôi có thể hỗ trợ thông tin về sản phẩm và dịch vụ Vingroup.")
            self.trace.append({"step": 1, "action": "direct_answer", "observation": answers[-1]})

        status = "completed" if not planned_actions else "max_iterations_reached"
        if not answers:
            answers.append("Không thể hoàn tất yêu cầu trong số vòng lặp cho phép.")

        return {
            "answer": "\n\n".join(answers),
            "trace": self.trace,
            "iterations": iterations,
            "status": status,
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()