import pandas as pd
import numpy as np
import random
import json
import sqlite3
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

print("Setup complete. Starting pipeline...")
# --- Step 1: Generate synthetic data ---
random.seed(42)

merchants = ["Zomato", "Swiggy", "Myntra", "BigBasket", "Nykaa", "Flipkart", "Ola", "Uber"]
merchant_variants = {
    "Zomato": "ZOMATO ONLINE SERVICES",
    "Swiggy": "SWIGGY BUNDL TECH",
    "Myntra": "MYNTRA DESIGNS",
}
base_date = datetime(2026, 8, 1)

gateway_records = []
settlement_records = []

for i in range(100):
    txn_id = f"TXN{1000+i}"
    utr = f"UTR{500000+i}"
    amount = round(random.uniform(100, 5000), 2)
    date = base_date + timedelta(days=random.randint(0, 25))
    merchant = random.choice(merchants)

    gateway_records.append({
        "transaction_id": txn_id, "utr_reference": utr, "amount": amount,
        "date": date.strftime("%Y-%m-%d"), "merchant": merchant, "true_label": None
    })

    scenario = random.random()

    if scenario < 0.65:
        settlement_records.append({
            "settlement_id": f"SETL{2000+i}", "utr_reference": utr, "amount": amount,
            "date": (date + timedelta(days=1)).strftime("%Y-%m-%d"), "merchant": merchant
        })
        gateway_records[-1]["true_label"] = "MATCH"
    elif scenario < 0.75:
        settlement_records.append({
            "settlement_id": f"SETL{2000+i}", "utr_reference": utr,
            "amount": round(amount - random.uniform(5, 25), 2),
            "date": (date + timedelta(days=1)).strftime("%Y-%m-%d"), "merchant": merchant
        })
        gateway_records[-1]["true_label"] = "NO_MATCH"
    elif scenario < 0.85:
        settlement_records.append({
            "settlement_id": f"SETL{2000+i}", "utr_reference": utr, "amount": amount,
            "date": (date + timedelta(days=random.randint(2, 4))).strftime("%Y-%m-%d"), "merchant": merchant
        })
        gateway_records[-1]["true_label"] = "MATCH"
    elif scenario < 0.93:
        settlement_records.append({
            "settlement_id": f"SETL{2000+i}", "utr_reference": utr, "amount": amount,
            "date": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
            "merchant": merchant_variants.get(merchant, merchant.upper())
        })
        gateway_records[-1]["true_label"] = "MATCH"
    else:
        gateway_records[-1]["true_label"] = "NO_MATCH"

gateway_df = pd.DataFrame(gateway_records)
settlement_df = pd.DataFrame(settlement_records)
gateway_df.to_csv("gateway_log.csv", index=False)
settlement_df.to_csv("bank_settlement.csv", index=False)

print(f"Generated {len(gateway_df)} gateway records, {len(settlement_df)} settlement records")

# --- Step 2: Prepare text and embeddings ---
gateway_df["match_text"] = (
    gateway_df["utr_reference"] + " " + gateway_df["merchant"] + " " + gateway_df["amount"].astype(str)
)
settlement_df["match_text"] = (
    settlement_df["utr_reference"] + " " + settlement_df["merchant"] + " " + settlement_df["amount"].astype(str)
)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

gateway_embeddings = model.encode(gateway_df["match_text"].tolist())
settlement_embeddings = model.encode(settlement_df["match_text"].tolist())

dimension = settlement_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(settlement_embeddings).astype("float32"))

print(f"Embeddings built. Dimension: {dimension}, Settlement records indexed: {index.ntotal}")

# --- Step 3: Confidence scoring and matching ---
def compute_confidence(embedding_distance, amount_diff, date_diff):
    embedding_score = max(0, 1 - embedding_distance)
    amount_score = max(0, 1 - (amount_diff / 100))
    date_score = max(0, 1 - (date_diff / 10))
    return round((0.5 * embedding_score) + (0.3 * amount_score) + (0.2 * date_score), 3)

def get_amount_threshold(amount):
    return max(10, amount * 0.01)

distances, indices = index.search(np.array(gateway_embeddings).astype("float32"), k=5)

candidate_list = []
for i, row in gateway_df.iterrows():
    candidate_list.append((i, row, distances[i], indices[i]))
candidate_list.sort(key=lambda x: x[2][0])

used_settlements = set()
final_results = []

for i, row, dist_row, idx_row in candidate_list:
    best_candidate = None
    best_confidence = -1

    for rank in range(5):
        candidate_idx = idx_row[rank]
        if candidate_idx in used_settlements or candidate_idx >= len(settlement_df):
            continue
        matched_settlement = settlement_df.iloc[candidate_idx]
        amount_diff = abs(row["amount"] - matched_settlement["amount"])
        date_diff = abs((pd.to_datetime(row["date"]) - pd.to_datetime(matched_settlement["date"])).days)
        confidence = compute_confidence(dist_row[rank], amount_diff, date_diff)
        if confidence > best_confidence:
            best_confidence = confidence
            best_candidate = (candidate_idx, matched_settlement, amount_diff, date_diff, confidence)

    if best_candidate is None or best_confidence < 0.4:
        final_results.append({
            "transaction_id": row["transaction_id"], "true_label": row["true_label"],
            "matched_settlement_id": None, "confidence": best_confidence if best_candidate else 0,
            "amount_diff": None, "date_diff_days": None, "status": "no_confident_match"
        })
    else:
        candidate_idx, matched_settlement, amount_diff, date_diff, confidence = best_candidate
        used_settlements.add(candidate_idx)
        amount_ok = amount_diff <= get_amount_threshold(row["amount"])
        status = "auto_confirmed" if (confidence >= 0.85 and amount_ok) else "needs_review"
        final_results.append({
            "transaction_id": row["transaction_id"], "true_label": row["true_label"],
            "matched_settlement_id": matched_settlement["settlement_id"], "confidence": confidence,
            "amount_diff": round(amount_diff, 2), "date_diff_days": date_diff, "status": status
        })

results_df = pd.DataFrame(final_results)
print(results_df["status"].value_counts())

# --- Step 4: LLM review for uncertain cases ---
review_df = results_df[results_df["status"] == "needs_review"].copy()

def get_llm_decision(row):
    gateway_row = gateway_df[gateway_df["transaction_id"] == row["transaction_id"]].iloc[0]
    settlement_row = settlement_df[settlement_df["settlement_id"] == row["matched_settlement_id"]].iloc[0]

    prompt = f"""You are a financial reconciliation reviewer. Decide if these two records represent the same transaction, using ONLY the evidence given. Never invent details not shown here.

Gateway record: UTR={gateway_row['utr_reference']}, Amount=₹{gateway_row['amount']}, Date={gateway_row['date']}, Merchant={gateway_row['merchant']}
Candidate settlement record: UTR={settlement_row['utr_reference']}, Amount=₹{settlement_row['amount']}, Date={settlement_row['date']}, Merchant={settlement_row['merchant']}
Amount difference: ₹{row['amount_diff']}, Date difference: {row['date_diff_days']} days, System confidence score: {row['confidence']}

Respond ONLY with valid JSON in this exact format, nothing else:
{{"decision": "MATCH" or "NO_MATCH" or "ESCALATE", "confidence": a number 0 to 1, "reason": "one short sentence"}}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except:
        return {"decision": "ESCALATE", "confidence": 0, "reason": "Could not parse LLM response"}

if len(review_df) > 0:
    llm_decisions = review_df.apply(get_llm_decision, axis=1)
    review_df["llm_decision"] = llm_decisions.apply(lambda x: x["decision"])
    review_df["llm_confidence"] = llm_decisions.apply(lambda x: x["confidence"])
    review_df["llm_reason"] = llm_decisions.apply(lambda x: x["reason"])

    def map_llm_status(decision):
        if decision == "MATCH":
            return "llm_confirmed"
        elif decision == "NO_MATCH":
            return "llm_rejected"
        else:
            return "needs_human_review"

    review_df["final_status"] = review_df["llm_decision"].apply(map_llm_status)

    results_df = results_df.merge(
        review_df[["transaction_id", "llm_decision", "llm_confidence", "llm_reason", "final_status"]],
        on="transaction_id", how="left"
    )
else:
    results_df["llm_decision"] = None
    results_df["llm_confidence"] = None
    results_df["llm_reason"] = None
    results_df["final_status"] = None

results_df["final_status"] = results_df["final_status"].fillna(results_df["status"])

print(f"LLM reviewed {len(review_df)} records")
print(results_df["final_status"].value_counts())

# --- Step 5: Metrics ---
results_df["system_final_decision"] = results_df["final_status"].apply(
    lambda s: "NO_MATCH" if s in ["no_confident_match", "llm_rejected"] else "MATCH"
)

true_positive = ((results_df["true_label"] == "MATCH") & (results_df["system_final_decision"] == "MATCH")).sum()
false_positive = ((results_df["true_label"] == "NO_MATCH") & (results_df["system_final_decision"] == "MATCH")).sum()
false_negative = ((results_df["true_label"] == "MATCH") & (results_df["system_final_decision"] == "NO_MATCH")).sum()

precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

auto_only = results_df[results_df["final_status"] == "auto_confirmed"]
auto_false_rate = (auto_only["true_label"] == "NO_MATCH").sum() / len(auto_only) if len(auto_only) > 0 else 0
llm_review_rate = len(review_df) / len(results_df)

print(f"\n--- FINAL METRICS ---")
print(f"Precision: {round(precision, 3)}")
print(f"Recall: {round(recall, 3)}")
print(f"Auto-confirm false match rate: {round(auto_false_rate, 3)}")
print(f"LLM review rate: {round(llm_review_rate, 3)}")

# --- Step 6: Audit trail ---
conn = sqlite3.connect("audit_trail.db")
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS reconciliation_log")
cursor.execute("""
CREATE TABLE reconciliation_log (
    transaction_id TEXT, matched_settlement_id TEXT, confidence REAL,
    amount_diff REAL, date_diff_days INTEGER, llm_decision TEXT,
    llm_confidence REAL, llm_reason TEXT, final_status TEXT,
    logged_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

for _, row in results_df.iterrows():
    cursor.execute("""
    INSERT INTO reconciliation_log
    (transaction_id, matched_settlement_id, confidence, amount_diff, date_diff_days, llm_decision, llm_confidence, llm_reason, final_status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["transaction_id"], row["matched_settlement_id"], row["confidence"],
        row["amount_diff"], row["date_diff_days"], row.get("llm_decision"),
        row.get("llm_confidence"), row.get("llm_reason"), row["final_status"]
    ))
conn.commit()
conn.close()

# --- Step 7: Export for dashboard ---
results_df.to_csv("final_results.csv", index=False)

print("\nPipeline complete. audit_trail.db and final_results.csv are ready.")