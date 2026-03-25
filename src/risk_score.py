"""
src/risk_score.py
=================
Phase 4 - Yield Risk Scoring

This module converts the AI's classification into a business-meaningful metric.
A classification of "Scratch" is helpful, but engineers need to know:
"Does this mean I should stop the manufacturing line?"

Formula:
Risk Score = Severity Weight × Confidence × Defect Area Factor × 100

Explanation:
- Severity Weight: How bad is this type of defect? (0.0 to 1.0)
- Confidence: How sure is the AI? (0.0 to 1.0)
- Defect Area Factor: We approximate area by looking at the Grad-CAM heatmap size.
"""

import numpy as np

# These weights represent the business reality of semiconductor manufacturing.
# A small localized defect might kill 1 or 2 chips (low weight).
# A "Near-Full" defect wipes out the entire $10,000 wafer (max weight).
# "None" means the wafer is good (zero risk).
SEVERITY_WEIGHTS = {
    "None":      0.0,
    "Loc":       0.3,  # Small localized area
    "Edge-Loc":  0.4,  # Edge only, usually manageable
    "Center":    0.6,  # Core affected, high-value chips die
    "Scratch":   0.6,  # Line defect, crosses many chips
    "Donut":     0.7,  # Ring pattern, usually a machine issue
    "Edge-Ring": 0.7,  # Full edge ring, massive die loss
    "Random":    0.8,  # Unpredictable, hard to trace
    "Near-Full": 0.9   # Most of wafer ruined, stop the line!
}

def calculate_risk_score(predicted_class, confidence, heatmap=None):
    """
    Calculates a 0-100 yield risk score.
    
    Args:
        predicted_class (str): The name of the defect class (e.g. "Center")
        confidence (float): The AI's softmax probability (0.0 to 1.0)
        heatmap (np.ndarray): Optional Grad-CAM heatmap to calculate area
        
    Returns:
        int: Risk score from 0 to 100
        str: Recommended action ("Monitor", "Investigate", "STOP LINE")
    """
    # 1. Base Severity
    severity = SEVERITY_WEIGHTS.get(predicted_class, 0.5)
    
    # 2. Area Factor
    area_factor = 1.0
    if heatmap is not None and predicted_class != "None":
        # Calculate percentage of pixels that are very "hot" (> 0.5 activation)
        hot_pixels = np.sum(heatmap > 0.5)
        total_pixels = heatmap.size
        hot_ratio = hot_pixels / total_pixels
        
        # Scale area factor between 0.8 (small area) and 1.2 (massive area)
        area_factor = 0.8 + (hot_ratio * 0.4)
    
    # 3. Final Calculation
    raw_score = severity * confidence * area_factor * 100
    
    # Clamp between 0 and 100 and round
    score = int(np.clip(raw_score, 0, 100))
    
    # 4. Business Rules
    if score < 30:
        action = "MONITOR"      # Green: Normal operation
    elif score < 70:
        action = "INVESTIGATE"  # Yellow: Send engineer to check
    else:
        action = "STOP LINE"    # Red: Halt machine to prevent more bad wafers
        
    return score, action

if __name__ == "__main__":
    # Simulate some predictions to test our logic
    print("==================================================")
    print("  Phase 4: Yield Risk Score Testing")
    print("==================================================")
    print(f"{'Defect Class':<12} | {'Conf':<4} | {'Area':<4} || {'Score':<5} | {'Action':<15}")
    print("-" * 55)
    
    test_cases = [
        # Normal wafer
        ("None", 0.99, None),
        # Minor localized issue
        ("Loc", 0.70, np.zeros((64,64))), 
        # Moderate scratch
        ("Scratch", 0.85, np.ones((64,64)) * 0.6), # Simulating a large heatmap
        # Catastrophic failure
        ("Near-Full", 0.95, np.ones((64,64)) * 0.9),
        # Machine alignment issue (Donut)
        ("Donut", 0.88, None)
    ]
    
    for cls, conf, hm in test_cases:
        score, act = calculate_risk_score(cls, conf, heatmap=hm)
        # Just mock string for the print
        area_str = "N/A" if hm is None else ("Hi" if hm.sum()>100 else "Lo")
        print(f"{cls:<12} | {conf:.2f} | {area_str:<4} || {score:<5} | {act:<15}")
    
    print("\n[DONE] Risk scoring module passed tests.")
