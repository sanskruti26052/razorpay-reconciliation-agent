import streamlit as st
import pandas as pd

st.set_page_config(page_title="Reconciliation Dashboard v2", layout="wide")

st.title("Multi-Source Reconciliation Agent")
st.caption("Matching payment gateway records against bank settlement records — with confidence scoring and LLM review")

results_df = pd.read_csv("final_results.csv")

total = len(results_df)
auto_confirmed = (results_df["final_status"] == "auto_confirmed").sum()
llm_confirmed = (results_df["final_status"] == "llm_confirmed").sum()
llm_rejected = (results_df["final_status"] == "llm_rejected").sum()
no_confident = (results_df["final_status"] == "no_confident_match").sum()
llm_reviewed_total = llm_confirmed + llm_rejected

# Real accuracy metrics
results_df["system_final_decision"] = results_df["final_status"].apply(
    lambda s: "NO_MATCH" if s in ["no_confident_match", "llm_rejected"] else "MATCH"
)
tp = ((results_df["true_label"] == "MATCH") & (results_df["system_final_decision"] == "MATCH")).sum()
fp = ((results_df["true_label"] == "NO_MATCH") & (results_df["system_final_decision"] == "MATCH")).sum()
fn = ((results_df["true_label"] == "MATCH") & (results_df["system_final_decision"] == "NO_MATCH")).sum()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

auto_df = results_df[results_df["final_status"] == "auto_confirmed"]
auto_false_rate = (auto_df["true_label"] == "NO_MATCH").sum() / len(auto_df) if len(auto_df) > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", total)
col2.metric("Auto-Confirmed", auto_confirmed)
col3.metric("LLM-Reviewed", llm_reviewed_total, f"{round(llm_reviewed_total/total*100,1)}% of all records")
col4.metric("No Confident Match", no_confident)

st.divider()

col5, col6, col7 = st.columns(3)
col5.metric("Precision", f"{round(precision*100,1)}%")
col6.metric("Recall", f"{round(recall*100,1)}%")
col7.metric("Auto-Confirm False Match Rate", f"{round(auto_false_rate*100,1)}%", help="Of records the system was fully confident about, how many were actually wrong")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["✅ Auto-Confirmed", "🤖 LLM-Confirmed", "🚫 LLM-Rejected", "❓ No Confident Match"])

with tab1:
    st.dataframe(results_df[results_df["final_status"] == "auto_confirmed"][
        ["transaction_id", "matched_settlement_id", "confidence", "amount_diff", "date_diff_days"]
    ], use_container_width=True)

with tab2:
    st.dataframe(results_df[results_df["final_status"] == "llm_confirmed"][
        ["transaction_id", "matched_settlement_id", "confidence", "llm_reason"]
    ], use_container_width=True)

with tab3:
    st.dataframe(results_df[results_df["final_status"] == "llm_rejected"][
        ["transaction_id", "matched_settlement_id", "confidence", "llm_reason"]
    ], use_container_width=True)

with tab4:
    st.subheader("No Confident Match Found")
    st.caption("No settlement candidate met the minimum confidence threshold — requires manual investigation.")
    st.dataframe(results_df[results_df["final_status"] == "no_confident_match"][
        ["transaction_id"]
    ], use_container_width=True)

st.divider()
st.caption("Built with sentence-transformers + FAISS for candidate matching, a weighted confidence score with a hard amount-safety gate, Groq (openai/gpt-oss-20b) for structured LLM review, and SQLite for the audit trail.")