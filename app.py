import streamlit as st
import pandas as pd
import json
import os
from db_access import *
from db_persist import persist_run, persist_audit_log
from llm_client import LLMClient

st.set_page_config(page_title="RefurbOS", layout="wide", page_icon="⌚")

# UI STYLING
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .grade-a { color: #00cc00; font-weight: bold; }
    .grade-d { color: #ff3333; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def show_dashboard():
    st.title("⌚ RefurbOS: Wearable Operations")
    
    # 1. LOAD DATA
    try: 
        devices = list_devices()
        if not devices: 
            st.error("Database Empty. Run seed_smartwatch.py")
            return
    except Exception as e:
        st.error(f"Connection Failed: {e}")
        return

    # 2. SIDEBAR SELECTOR
    dev_map = {d['device_id']: d for d in devices}
    sel_id = st.sidebar.selectbox("Select Unit", list(dev_map.keys()))
    data = get_device_payload(sel_id)
    
    # Unpack for easy access
    specs = data.get('specs', {})
    cond = data.get('condition', {})
    tel = data.get('telemetry', {})
    
    # 3. HEADER METRICS (No N/A allowed)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", f"{data.get('brand')} {data.get('model')}")
    c2.metric("Size", specs.get('case_size', 'Unknown'))
    c3.metric("Battery", f"{cond.get('battery_health', 0)}%")
    
    grade = cond.get('grade', 'Unknown')
    color = "grade-a" if "Grade A" in grade else "grade-d"
    c4.markdown(f"<div class='metric-box'><span class='{color}'>{grade}</span></div>", unsafe_allow_html=True)

    # 4. TABS
    t1, t2, t3 = st.tabs(["🔍 Deep Dive", "🤖 AI Tribunal", "📜 Chain of Custody"])
    
    # TAB 1: VISUAL INSPECTION
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("PHYSICAL STATUS")
            st.write(f"**Band:** {specs.get('band', 'Standard')}")
            st.write(f"**Material:** {specs.get('material', 'Standard')}")
            st.write(f"**Scratches:** {cond.get('scratches', 'None')}")
        with c2:
            st.caption("SENSOR TELEMETRY")
            st.write(f"**Oxidation:** {tel.get('sensor_oxidation', 'Clean')}")
            st.write(f"**Haptics:** {tel.get('haptic_response', 'Normal')}")
            
            # Simulated Chart
            st.line_chart([10, 12, 10, 11, 40, 10, 12] if tel.get('haptic_response') == "Weak" else [10, 10, 11, 10, 10, 11, 10])

    # TAB 2: TRIBUNAL (Live AI)
    with t2:
        if st.button("Start Evaluation"):
            llm = LLMClient("openai") # or gemini
            rid = f"RUN-{pd.Timestamp.now().strftime('%H%M%S')}"
            persist_run(rid, sel_id)
            
            p1 = f"Review {data}. concise risk assessment."
            r1 = llm.complete(p1)
            st.info(f"RISK ASSESSMENT: {r1}")
            persist_audit_log(rid, "AI_RISK_ASSESSMENT", {"summary": r1[:100] + "..."})

            p2 = f"Review {r1}. Estimated Resale Value?"
            r2 = llm.complete(p2)
            st.success(f"VALUATION: {r2}")
            persist_audit_log(rid, "AI_VALUATION", {"value_prediction": r2})

    # TAB 3: AUDIT LOG (Clean Table, No Chat)
    with t3:
        runs = fetch_runs_for_device(sel_id)
        if not runs.empty:
            all_logs = []
            for rid in runs['run_id']:
                chain = fetch_audit_chain(rid)
                for _, row in chain.iterrows():
                    # Clean up the JSON for display
                    details = row['payload_json']
                    if isinstance(details, str): details = json.loads(details)
                    
                    all_logs.append({
                        "Time": row['created_at'],
                        "Event": row['event_type'],
                        "Details": str(details) # simplified string
                    })
            
            if all_logs:
                st.dataframe(pd.DataFrame(all_logs), use_container_width=True)
            else:
                st.info("Device initialized. No events yet.")

if __name__ == "__main__":
    show_dashboard()
