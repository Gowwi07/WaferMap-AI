# Phase 4: Yield Risk Scoring

**Goal:** Convert the AI's technical classification into a simple, actionable business metric (Yield Risk Score, 0–100) so engineers know if they need to stop the manufacturing line.

An AI saying "This is an Edge-Ring defect with 92% confidence" is great, but a factory manager wants to know: "How much money am I losing right now, and should I halt the machine?"

In this phase, we implemented the business logic in [src/risk_score.py](file:///e:/Projects/SanDisk/src/risk_score.py) to answer that question.

---

## 🧮 1. The Risk Score Formula

We created a formula that mimics real-world semiconductor manufacturing logic:

`Risk Score = Severity × Area Factor × AI Confidence × 100`

### A. Severity (0.0 to 1.0)
Not all defects are equal. A tiny speck of dust is cheaper to ignore than a massive scratch that ruins 50 high-end processor chips. We assigned a base weight to each of the 9 classes:

| Defect Type | Severity | Why? |
|-------------|----------|------|
| **None** | `0.0` | Normal wafer, zero risk. |
| **Loc** | `0.3` | Small localized drop. Usually only ruins 1 or 2 chips. |
| **Edge-Loc** | `0.4` | Edge chips are often test structures anyway. Low risk. |
| **Center / Scratch** | `0.6` | The center is where the most valuable chips are. Scratches cross multiple chips. |
| **Donut / Edge-Ring**| `0.7` | Distinct patterns usually indicate a specific machine is misaligned or has a bad seal. High risk. |
| **Random** | `0.8` | The worst kind of defect is one you can't find the root cause for. |
| **Near-Full** | `0.9` | The entire $10,000 wafer is ruined. Immediate action required. |

### B. Area Factor (0.8x to 1.2x)
We use the **Grad-CAM heatmap** from Phase 3! 
If the "hot" area of the heatmap covers a huge portion of the wafer, we multiply the risk by `1.2` (making it worse). If the heatmap only highlights a tiny dot, we multiply by `0.8` (lowering the risk).

### C. AI Confidence (0.0 to 1.0)
If the AI is only 55% sure it's a "Donut" pattern, we lower the risk score compared to if the AI is 99% sure.

---

## 🚦 2. Business Actions

The final Risk Score (0-100) is tied to three actionable states:

🟢 **MONITOR (Score 0-29)**
* **Meaning:** Everything is fine, or minor anomalies exist that don't threaten overall yield.
* **Action:** Keep the machines running.

🟡 **INVESTIGATE (Score 30-69)**
* **Meaning:** Moderate defects detected. Yield is dropping.
* **Action:** Send a process engineer to inspect the machine at the end of its current run.

🔴 **STOP LINE (Score 70-100)**
* **Meaning:** Catastrophic failure or severe systemic issue detected.
* **Action:** Halt the manufacturing equipment immediately to prevent ruining subsequent expensive wafers.

---

## ✅ Phase Complete!

We now have all the independent pieces of our AI system:
1. **Phase 1 Data:** Wafer handling
2. **Phase 2 AI:** EfficientNet Classification
3. **Phase 3 Explainability:** Grad-CAM Localization
4. **Phase 4 Business Logic:** Risk Scoring

**Next Up — Phase 5:** We will stitch all these scripts together into a beautiful, interactive **Streamlit Dashboard** that you can use in your browser!
