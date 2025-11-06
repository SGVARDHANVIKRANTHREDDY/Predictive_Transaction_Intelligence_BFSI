# =============================================================================
# fraud_rules.py — Simple Rule-Based Explainability Helper
# =============================================================================

def evaluate_rules(transaction):
    """
    Apply business rules to flag suspicious transactions.
    Returns a list of triggered rule explanations.
    """
    rules_triggered = []

    if transaction.get("amount", 0) > 100000:
        rules_triggered.append("High transaction amount")

    if transaction.get("country", "").lower() not in ["india", "usa", "uk"]:
        rules_triggered.append("Foreign transaction")

    if int(transaction.get("hour", 12)) < 6 or int(transaction.get("hour", 12)) > 23:
        rules_triggered.append("Unusual transaction time (night)")

    if transaction.get("device_type", "").lower() == "unknown":
        rules_triggered.append("Unrecognized device")

    return rules_triggered
