# Counterfactual collection policy snapshot

`brl/jsp/policy.py` is the exact policy file used by both final counterfactual collections. Its SHA-256 is `5f6822e088803c16ecea4a3bf17cd8bbc871ed56779dc9adf69ae99b2dd830cd`, matching both collection manifests.

The current repository policy differs only by three telemetry lines that count optical fallback entries and expose that count in the result. Candidate generation, features, action selection, execution, completion and branch costs are unchanged. The final sequential development manifests hash the current telemetry-enabled file.
