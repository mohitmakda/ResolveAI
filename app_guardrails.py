"""Account-specific status detection, independent of Streamlit."""
import re

def requires_account_verification(question):

    text = re.sub(r"\s+", " ", question.lower().strip())
    clauses = re.split(r"(?<=[.!?])\s+", text)

    guidance = [
        r"^(?:what should|how should|what can|how can|how do|how to)\b",
        r"^(?:what happens|what is the process|what are the steps)\b",
        r"^(?:can|could) you (?:explain|describe)\b",
        r"^(?:can|could) you confirm (?:how|the steps|the procedure)\b",
        r"^(?:can|could) you tell me how\b",
        r"\b(?:normally|typically|usually)\b",
    ]

    status = [
        r"^(?:has|have|had|did|was|were|is|are)\b",
        r"^(?:where is|where's)\s+my\b",
        r"^when will my\b",
        r"^why was my\b",
        r"^(?:what|which)(?: is| was)? (?:the )?exact\b",
        r"^what is my\b",
        r"\b(?:has|have|was|is)\s+my\s+(?:refund|reversal|transfer|payment|withdrawal|chargeback|card|account)\b",
        r"\b(?:status|balance|tracking number|arrival date|refunded|credited|reversed|delivered|processed)\b.{0,55}\b(?:my|mine)\b",
        r"^(?:can|could|would) you (?:tell me my|tell me if|confirm (?:that |whether )?my)\b",
        r"^(?:can|could|would) you confirm (?:that|whether)\b",
    ]

    for clause in clauses:
        clause = clause.strip()
        if any(re.search(p, clause) for p in guidance):
            continue
        if any(re.search(p, clause) for p in status):
            return True

    return False
