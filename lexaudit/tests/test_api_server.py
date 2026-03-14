from __future__ import annotations

from fastapi.testclient import TestClient

from src.agent.audit import AuditLogger
from src.agent.control_loop import _finalize_result
from src.api import server
from src.config import Settings
from src.types import AgentState


def test_continue_endpoint_finalizes_pending_review(monkeypatch, tmp_path) -> None:
    class _StubWeilAuditLogger:
        enabled = True
        wallet_address = "wallet-123"

        def __init__(self) -> None:
            self.tx_results = []

        def emit(self, event_type: str, data: dict, *, wait_for_confirmation: bool = False) -> None:
            if event_type == "AUDIT_COMPLETE":
                self.tx_results.append(
                    {
                        "event_type": event_type,
                        "status": "TransactionStatus.FINALIZED",
                        "block_height": 101,
                        "batch_id": "batch-api-789",
                        "tx_idx": 2,
                        "tx_hash": None,
                    }
                )
                return
            self.tx_results.append(
                {
                    "event_type": event_type,
                    "status": "TransactionStatus.IN_PROGRESS",
                    "block_height": 0,
                    "batch_id": "",
                    "tx_idx": 0,
                    "tx_hash": None,
                }
            )

    state = AgentState(
        contract_text="Contract text",
        filename="contract.txt",
        session_id="sess-api",
        human_decision="pending",
        terminate_reason="HUMAN_REVIEW_PENDING",
        human_gate_open=True,
    )
    audit = AuditLogger(tmp_path, "sess-api")
    pending_result = _finalize_result(state, audit, _StubWeilAuditLogger(), pending_human_review=True)

    monkeypatch.setattr(
        server,
        "load_settings",
        lambda: Settings(
            anthropic_api_key="",
            runs_dir=tmp_path,
            weilchain_pod_url="https://marauder.weilliptic.ai",
        ),
    )
    monkeypatch.setattr(server, "_pending_audit_result", pending_result)
    monkeypatch.setattr(server, "_pending_result", pending_result.to_dict())

    client = TestClient(server.app)
    response = client.post("/api/continue", json={"decision": "reject"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["pending_human_review"] is False
    assert payload["state"]["human_decision"] == "reject"
    assert payload["chain_status"] == "TransactionStatus.FINALIZED"
    assert payload["batch_id"] == "batch-api-789"
    assert payload["chain_reference"] == "batch-api-789"
    assert server._pending_result == {}
    assert server._pending_audit_result is None
