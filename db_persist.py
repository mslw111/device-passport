import json
from db_access import get_conn

def persist_run(rid, did):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO runs (run_id, device_id) VALUES (%s, %s)", (rid, did))
            conn.commit()

def persist_audit_log(rid, etype, payload):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)", 
                        (rid, etype, json.dumps(payload)))
            conn.commit()
