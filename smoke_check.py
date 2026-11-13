"""Smoke test that needs only base requirements (no Streamlit import)."""
import json
from pathlib import Path
from core import POLICIES, retrieve_v3, policy_supported, train_v3, BANKING77_PATH
from reliability import check_domain, input_risk
root=Path(__file__).resolve().parent
assert len(POLICIES)>=15
assert all(isinstance(p.get("id"),str) and p.get("content") for p in POLICIES)
query="I transferred money to another bank account yesterday but the recipient has not received it."
docs=retrieve_v3(query,None,k=3,semantic=False)
assert docs and docs[0]["id"]=="POL-05", docs
assert check_domain("What is the weather today?")["in_domain"] is False
assert input_risk("I shared my banking password with someone") == "credentials_exposed"
model,stats=train_v3(False)
print(json.dumps({"policies":len(POLICIES), "transfer_policy":docs[0]["id"], "demo_model_classes":stats["classes"], "banking77_present":BANKING77_PATH.exists()}, indent=2))
print("SMOKE CHECK PASS")
