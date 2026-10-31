# Security guidance

This repo demonstrates a **synthetic** credit-risk dataset; it is not a production banking platform.

- Keep `ANTHROPIC_API_KEY` out of Git; set it using deployment secrets.
- To enable anomaly review decisions, configure a long random `REVIEW_WRITE_TOKEN` in Streamlit Secrets or the environment. Without it, review writes are disabled. This shared token is only a safeguard for demos, **not** a substitute for SSO, RBAC, protected audit logs and server-side sessions in production.
- Generated SQL is untrusted: the SQLite authorizer restricts tables and functions and query execution is bounded. Always review supporting SQL/evidence before acting on results.
- Do not publish real customer data or send identifiable data to an external LLM without lawful authorization and privacy safeguards.
- Run `python -m pytest tests -q`, `pip-audit -r requirements.txt` and `bandit -r app src -ll -x tests` before deployment.
- Private deployments should require identity-aware access control; a public dashboard is not suitable for customer financial records.
- If any live secret was leaked into Git, revoke/rotate it; deleting a tracked file does not erase Git history.
