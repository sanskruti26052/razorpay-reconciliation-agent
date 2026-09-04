
# Multi-Source Reconciliation Agent

An AI-assisted reconciliation system that matches payment gateway transactions against bank settlement records, using confidence-based scoring and LLM review for ambiguous cases — built for the Razorpay AI Builder Internship 2026 (AI Finance Controller track).

## Problem

When a customer pays through a payment gateway, that transaction is later settled into the merchant's bank account — but the two records (gateway log vs. bank settlement) don't always match perfectly. Amounts can differ due to fees, settlements can be delayed, merchant names can be recorded differently across systems, and some transactions may never settle at all. Today, this reconciliation is largely manual — slow, error-prone, and doesn't scale.

This project builds an agent that automates that matching, flags genuine discrepancies, and produces an auditable trail of every decision it makes.

## Architecture

The pipeline works in six stages:

1. **Data ingestion** — Two synthetic datasets are generated: a payment gateway log and a bank settlement file, deliberately including realistic discrepancies (amount mismatches, delayed settlements, merchant name variants, and missing settlements).
2. **Embedding + candidate retrieval** — Each record's key fields (UTR reference, merchant, amount) are converted into vector embeddings using `sentence-transformers`, and FAISS is used to retrieve the top candidate matches for each gateway record.
3. **Confidence scoring** — Each candidate is scored on a weighted combination of embedding similarity, amount closeness, and date closeness. A hard rule caps the amount tolerance (1% or ₹10, whichever is greater) — a candidate can never be auto-confirmed if the amount doesn't fall within this tolerance, regardless of how similar everything else looks.
4. **Decision routing** — Based on the confidence score:
   - **High confidence + amount within tolerance** → auto-confirmed, no review needed
   - **Uncertain** → routed to an LLM for review
   - **No candidate clears the minimum confidence threshold** → flagged as "no confident match found" (not assumed unsettled — flagged for investigation)
5. **LLM review** — Uncertain cases are sent to Groq (`openai/gpt-oss-20b`) with structured evidence. The LLM returns a structured decision (MATCH / NO_MATCH / ESCALATE) with a reason, and this decision directly updates the transaction's final status.
6. **Audit trail + dashboard** — Every decision (confidence score, amount/date differences, LLM reasoning where applicable, final status) is logged to a SQLite database, and a Streamlit dashboard displays the results with real precision/recall metrics.

**Stack:** Python, pandas, sentence-transformers, FAISS, Groq (LLM), SQLite, Streamlit — entirely free-tier tools.

## Metrics

Since the synthetic dataset is generated with known ground truth (each record is labeled MATCH or NO_MATCH at creation), the system's real accuracy can be measured directly rather than estimated.

| Metric | Value |
|---|---|
| Precision | 91.2% |
| Recall | 96.5% |
| Auto-confirm false match rate | 6.7% |
| LLM review rate | 3% (only 3 of 100 records needed AI review) |

**Auto-confirm false match rate** is the most important number here — it measures how often the system's *most confident* decisions (auto-confirmed, no review) were actually wrong. In an earlier version of this system, this number was 64%, because the confidence formula weighted text similarity too heavily relative to amount accuracy. Adding a hard amount-tolerance rule brought this down to 6.7%.

**LLM review rate** shows the system is not calling an LLM on every record — only the genuinely ambiguous ~3% require it, keeping the pipeline fast and cheap while reserving AI reasoning for cases that actually need judgment.

## How to Run

**1. Clone the repository and enter the folder**
```bash
git clone https://github.com/sanskruti26052/razorpay-reconciliation-agent.git
cd razorpay-reconciliation-agent
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Copy `.env.example` to a new file named `.env`, and add your own free Groq API key (get one at [console.groq.com](https://console.groq.com)):

GROQ_API_KEY=your_key_here

**5. Run the pipeline**
```bash
python pipeline.py
```
This generates the synthetic data, runs the full matching + scoring + LLM review pipeline, and produces `final_results.csv` and `audit_trail.db`.

**6. Launch the dashboard**
```bash
streamlit run dashboard.py
```
Opens a local dashboard at `http://localhost:8501` showing the results and metrics.

## Limitations & Future Work

This is a focused prototype, not a production system.
Known limitations:

- **One-directional matching only** — the system checks gateway → settlement (does every payment have a matching settlement?), but does not currently check the reverse: settlement records with no corresponding gateway transaction, which in a real system could indicate a bank-side error or unauthorized credit.
- **No partial settlement handling** — a single gateway transaction settled across multiple smaller bank entries (e.g. after a partial refund) is not currently detected.
- **No duplicate transaction detection** — exact or near-duplicate transactions are not explicitly flagged.
- **Greedy matching, not globally optimal** — candidates are assigned in order of confidence, which can occasionally produce a locally reasonable but not globally optimal assignment across the full batch.
- **Small-scale demo** — tested on 100 synthetic records; a production system would need to be stress-tested at much larger volumes to validate matching speed and accuracy at scale.
- **LLM review has a shared blind spot with the confidence scorer** — both can occasionally over-trust strong UTR/merchant similarity even when the amount is genuinely wrong; a stricter policy gate on top of the LLM's decision would reduce this further.

With more time, the priority additions would be: bidirectional reconciliation, partial-settlement matching, and a human-in-the-loop approval step for `needs_human_review` cases before they're finalized.

## Why This Fits the AI Finance Controller Track

This project directly addresses the track's stated bar: **throughput plus measured accuracy plus an honest exception list.**

- **Throughput** — the full 100-record batch (data generation through audit logging) runs end-to-end with a single command.
- **Measured accuracy** — precision, recall, and a specific false-match-rate metric are computed against known ground truth, not estimated.
- **Honest exception list** — records the system isn't confident about are explicitly flagged as "no confident match found" rather than force-matched, and every decision (including the LLM's reasoning) is logged in a full audit trail.