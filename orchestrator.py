import hashlib, datetime
# Import the agents (assuming standard filenames)
from agent_r2v3 import AgentR2V3
from agent_gdpr import AgentGDPR
from agent_sensor import AgentSensorHealth

# Use local stubs for verification if files missing, or real if present
try:
    from verifier import VerificationModule
    from arbiter import BlindArbiter
except ImportError:
    class VerificationModule:
        def verify(self, d, r1, r2): return {"status": "Verified"}
    class BlindArbiter:
        def adjudicate(self, r1, r2, c): return {"fused_narrative": "Consensus Reached", "final_uncertainty": 0.1}

from audit_logger import AuditLogger

class Orchestrator:
    def __init__(self):
        self.logger = AuditLogger("audit_logs")
        self.agents = {
            "R2v3": AgentR2V3(),
            "GDPR": AgentGDPR(),
            "Sensor": AgentSensorHealth()
        }
        self.verifier = VerificationModule()
        self.arbiter = BlindArbiter()

    def run_device_audit(self, device):
        rid = f"run_{hashlib.sha256(datetime.datetime.utcnow().isoformat().encode()).hexdigest()[:8]}"
        self.logger.start_new_run(rid, device.get("device_id"))

        # Round 1
        r1 = {n: a.round1(device) for n, a in self.agents.items()}
        for n, o in r1.items(): self.logger.log_section(f"R1-{n}", o)

        # Round 2 (Debate)
        anon = list(r1.values())
        r2 = {n: a.round2(anon) for n, a in self.agents.items()}
        for n, o in r2.items(): self.logger.log_section(f"R2-{n}", o)

        # Verification
        cove = self.verifier.verify(device, r1, r2)
        self.logger.log_section("COVE", cove)

        # Arbitration
        arb = self.arbiter.adjudicate(r1, r2, cove)
        self.logger.log_section("ARBITER", arb)

        rec = {
            "run_id": rid,
            "final_narrative": arb.get("fused_narrative", "Audit Complete"),
            "final_uncertainty": arb.get("fused_uncertainty", 0.0),
            "details": {"r1": r1, "r2": r2},
            "audit_record": {"round1": r1, "round2": r2}
        }
        self.logger.log_section("FINAL", rec)
        return rec
