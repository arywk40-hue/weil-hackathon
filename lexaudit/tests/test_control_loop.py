from __future__ import annotations

import pytest

from src.agent.audit import AuditLogger
from src.agent.control_loop import _finalize_result, continue_human_review, run_lexaudit
from src.config import Settings
from src.tools.router import McpUnavailableError, ToolRouter, ToolSpec
from src.types import AgentState
from tests.conftest import InMemoryMCPClient, make_happy_path_handlers, make_high_risk_handlers


def _settings(tmp_path):
    return Settings(
        anthropic_api_key="",
        runs_dir=tmp_path,
        enforce_mcp=True,
        max_retries=1,
        retry_backoff_seconds=0.0,
    )


def test_happy_path_auto_approve(tmp_path) -> None:
    handlers = make_happy_path_handlers()
    router = ToolRouter(
        settings=_settings(tmp_path),
        mcp_client=InMemoryMCPClient(handlers),
    )

    result = run_lexaudit(
        "Contract text",
        "contract.txt",
        settings=_settings(tmp_path),
        router=router,
    )

    assert not result.state.fatal_error
    assert result.state.human_decision == "auto-approved"
    assert result.report_json["summary"]["low"] == 2


def test_human_gate_pending_when_high_risk(tmp_path) -> None:
    handlers = make_high_risk_handlers()
    router = ToolRouter(
        settings=_settings(tmp_path),
        mcp_client=InMemoryMCPClient(handlers),
    )

    result = run_lexaudit(
        "Contract text",
        "contract.txt",
        settings=_settings(tmp_path),
        router=router,
        decision_provider=None,
    )

    assert result.pending_human_review
    assert result.state.human_decision == "pending"
    assert result.state.terminate_reason == "HUMAN_REVIEW_PENDING"


def test_human_gate_pending_when_many_medium_risks(tmp_path) -> None:
    def clause_extractor(payload: dict) -> dict:
        return {
            "clauses": [
                {"id": idx, "title": f"Clause {idx}", "text": f"Medium-risk clause {idx}"}
                for idx in range(1, 9)
            ]
        }

    def risk_scorer(payload: dict) -> dict:
        return {
            "risk": {
                "risk_level": "MEDIUM",
                "confidence": "0.7",
                "reason": "Needs legal review",
                "flags": [],
            }
        }

    router = ToolRouter(
        settings=_settings(tmp_path),
        mcp_client=InMemoryMCPClient(
            {"clause_extractor": clause_extractor, "risk_scorer": risk_scorer}
        ),
    )

    result = run_lexaudit(
        "Contract text",
        "contract.txt",
        settings=_settings(tmp_path),
        router=router,
        decision_provider=None,
    )

    assert result.pending_human_review is True
    assert result.state.human_decision == "pending"
    assert result.state.terminate_reason == "HUMAN_REVIEW_PENDING"
    gate_event = next(event for event in result.audit_log if event.event_type == "HUMAN_GATE_OPEN")
    assert gate_event.metadata["flagged_clause_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert "medium_count>=8" in gate_event.metadata["policy_triggers"]
    assert gate_event.metadata["risk_counts"] == {"high": 0, "medium": 8, "low": 0}


def test_mcp_unavailable_fail_closed(tmp_path) -> None:
    class _UnavailableClient:
        def is_available(self) -> bool:
            return False

        def discover_tools(self) -> dict:
            return {}

        def call_tool(self, *, tool_name: str, method_name: str, payload: dict,
                      timeout_seconds: float, tool_spec: ToolSpec) -> dict:
            raise RuntimeError("unreachable")

    router = ToolRouter(settings=_settings(tmp_path), mcp_client=_UnavailableClient())
    with pytest.raises(McpUnavailableError):
        run_lexaudit(
            "Contract text",
            "contract.txt",
            settings=_settings(tmp_path),
            router=router,
            human_gate_enabled=False,
        )


def test_parse_retry_then_fail(tmp_path) -> None:
    def clause_extractor(payload: dict) -> dict:
        return {"clauses": [{"id": 1, "title": "Clause", "text": "X"}]}

    def risk_scorer(payload: dict) -> dict:
        return {
            "risk": {
                "risk_level": "CRITICAL",
                "confidence": "0.2",
                "reason": "invalid",
                "flags": [],
            }
        }

    router = ToolRouter(
        settings=_settings(tmp_path),
        mcp_client=InMemoryMCPClient({"clause_extractor": clause_extractor, "risk_scorer": risk_scorer}),
    )

    result = run_lexaudit(
        "Contract text",
        "contract.txt",
        settings=_settings(tmp_path),
        router=router,
        human_gate_enabled=False,
    )

    assert result.state.fatal_error
    assert result.state.terminate_reason == "INVALID_TOOL_OUTPUT"


def test_step_budget_termination(tmp_path) -> None:
    """Force loop to hit max_steps limit."""
    handlers = make_happy_path_handlers()
    router = ToolRouter(
        settings=_settings(tmp_path),
        mcp_client=InMemoryMCPClient(handlers),
    )

    result = run_lexaudit(
        "Contract text",
        "contract.txt",
        settings=_settings(tmp_path),
        router=router,
        max_steps=1,  # Force immediate termination
        human_gate_enabled=False,
    )

    assert result.state.terminate_reason == "MAX_STEPS_EXCEEDED"


def test_audit_events_emitted_at_every_node(tmp_path) -> None:
    """Run full pipeline on sample contract and verify audit events."""
    handlers = make_happy_path_handlers()
    router = ToolRouter(
        settings=_settings(tmp_path),
        mcp_client=InMemoryMCPClient(handlers),
    )

    result = run_lexaudit(
        "Contract text",
        "contract.txt",
        settings=_settings(tmp_path),
        router=router,
        human_gate_enabled=False,
    )

    # Check that key events exist
    event_types = [e.event_type for e in result.audit_log]
    assert "INIT" in event_types
    assert "INGEST_START" in event_types
    assert "TERMINATE" in event_types

    # Each event should have step, timestamp, and data
    for event in result.audit_log:
        assert event.step_index >= 0
        assert event.timestamp > 0
        assert isinstance(event.metadata, dict)


def test_finalize_result_surfaces_batch_id_when_sdk_has_no_tx_hash(tmp_path) -> None:
    class _StubWeilAuditLogger:
        enabled = True
        wallet_address = "wallet-123"

        def __init__(self) -> None:
            self.tx_results = []
            self.emit_calls = []

        def emit(self, event_type: str, data: dict, *, wait_for_confirmation: bool = False) -> None:
            self.emit_calls.append((event_type, wait_for_confirmation))
            if event_type == "AUDIT_COMPLETE":
                self.tx_results.append(
                    {
                        "event_type": event_type,
                        "status": "TransactionStatus.CONFIRMED",
                        "block_height": 77,
                        "batch_id": "batch-abc-123",
                        "tx_idx": 0,
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
                    "tx_idx": None,
                    "tx_hash": None,
                }
            )

    state = AgentState(
        contract_text="Contract text",
        filename="contract.txt",
        session_id="sess-123",
        human_decision="auto-approved",
        terminate_reason="complete",
    )
    audit = AuditLogger(tmp_path, "sess-123")
    weil_audit = _StubWeilAuditLogger()

    result = _finalize_result(state, audit, weil_audit, pending_human_review=False)
    payload = result.to_dict()

    assert ("AUDIT_COMPLETE", True) in weil_audit.emit_calls
    assert result.final_tx_hash is None
    assert result.final_chain_status == "TransactionStatus.CONFIRMED"
    assert result.final_batch_id == "batch-abc-123"
    assert result.final_chain_reference == "batch-abc-123"
    assert result.final_chain_reference_type == "batch_id"
    assert result.report_json["batch_id"] == "batch-abc-123"
    assert result.report_json["chain_reference"] == "batch-abc-123"
    assert payload["chain_reference"] == "batch-abc-123"
    assert payload["chain_reference_type"] == "batch_id"
    assert payload["block_height"] == 77


def test_continue_human_review_finalizes_pending_session(tmp_path) -> None:
    class _StubWeilAuditLogger:
        enabled = True
        wallet_address = "wallet-123"

        def __init__(self) -> None:
            self.tx_results = []
            self.emit_calls = []

        def emit(self, event_type: str, data: dict, *, wait_for_confirmation: bool = False) -> None:
            self.emit_calls.append((event_type, wait_for_confirmation))
            if event_type == "AUDIT_COMPLETE":
                self.tx_results.append(
                    {
                        "event_type": event_type,
                        "status": "TransactionStatus.FINALIZED",
                        "block_height": 88,
                        "batch_id": "batch-final-456",
                        "tx_idx": 1,
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
        session_id="sess-continue",
        human_decision="pending",
        terminate_reason="HUMAN_REVIEW_PENDING",
        human_gate_open=True,
    )
    audit = AuditLogger(tmp_path, "sess-continue")
    weil_audit = _StubWeilAuditLogger()

    pending = _finalize_result(state, audit, weil_audit, pending_human_review=True)
    continued = continue_human_review(pending, "reject")

    assert pending.pending_human_review is True
    assert ("HUMAN_DECISION", False) in weil_audit.emit_calls
    assert ("AUDIT_COMPLETE", True) in weil_audit.emit_calls
    assert continued.pending_human_review is False
    assert continued.state.human_decision == "reject"
    assert continued.state.human_gate_open is False
    assert continued.state.terminate_reason == "complete"
    assert continued.final_chain_status == "TransactionStatus.FINALIZED"
    assert continued.final_batch_id == "batch-final-456"
    assert continued.final_chain_reference == "batch-final-456"
    assert continued.report_json["chain_status"] == "TransactionStatus.FINALIZED"
    assert continued.report_json["batch_id"] == "batch-final-456"


def test_human_gate_reject_path(tmp_path) -> None:
    """HIGH risk contract + decision = 'reject'."""
    handlers = make_high_risk_handlers()
    router = ToolRouter(
        settings=_settings(tmp_path),
        mcp_client=InMemoryMCPClient(handlers),
    )

    # Decision provider returns "reject"
    def reject_provider(risk_scores):
        return "reject"

    result = run_lexaudit(
        "Contract text",
        "contract.txt",
        settings=_settings(tmp_path),
        router=router,
        decision_provider=reject_provider,
    )

    assert result.state.human_decision == "reject"
    # Check that HUMAN_DECISION appears in audit log metadata
    human_events = [e for e in result.audit_log if "decision" in e.metadata or "HUMAN" in e.event_type]
    assert len(human_events) > 0
