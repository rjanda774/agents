"""
Run this from your 6_mcp directory to fully reset Cathie's options account.
Clears all positions AND resets cash to $10,000.
Usage: python reset_cathie_options.py
"""
import sqlite3
import json

DB = "accounts.db"

with sqlite3.connect(DB) as conn:
    cur = conn.cursor()
    cur.execute("SELECT name FROM accounts")
    print("Accounts found:", [r[0] for r in cur.fetchall()])

    # Reset cathie_options (positions + collected premium)
    empty_options = {
        "name": "cathie_options",
        "open_positions": [],
        "closed_positions": [],
        "total_premium_collected": 0.0,
        "total_realized_pnl": 0.0,
        "cash": 10000.0
    }
    cur.execute("DELETE FROM accounts WHERE name = 'cathie_options'")
    cur.execute("INSERT INTO accounts (name, account) VALUES (?, ?)",
                ("cathie_options", json.dumps(empty_options)))

    # Reset cathie stock account cash to $10,000 (keeps strategy intact)
    cur.execute("SELECT account FROM accounts WHERE name = 'cathie'")
    row = cur.fetchone()
    if row:
        cathie = json.loads(row[0])
        cathie["balance"] = 10000.0
        cathie["holdings"] = {}
        cathie["transactions"] = []
        cathie["portfolio_value_time_series"] = []
        cur.execute("UPDATE accounts SET account = ? WHERE name = 'cathie'",
                    (json.dumps(cathie),))
        print("✓ cathie stock account reset — cash $10,000, holdings cleared")
    else:
        print("! cathie stock account not found, skipping")

    conn.commit()
    print("✓ cathie_options reset — positions cleared, cash $10,000")

print("Done. Restart the app.")
