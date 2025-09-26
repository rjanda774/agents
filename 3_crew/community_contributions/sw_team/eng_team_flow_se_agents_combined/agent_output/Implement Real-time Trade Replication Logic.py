To develop a system that allows users to replicate trades from traders they follow in real-time, we need to design a backend service that communicates with a trading engine and notifies users about trade actions. The solution should be modular, testable, and maintainable. Here's a Python implementation using object-oriented principles and design patterns:

```python
import threading
import time
from typing import List, Dict, Callable

class Trade:
    def __init__(self, trader_id: str, trade_details: Dict):
        self.trader_id = trader_id
        self.trade_details = trade_details

class TradingEngine:
    def execute_trade(self, trade: Trade):
        # Simulate trade execution
        print(f"Executing trade for trader {trade.trader_id}: {trade.trade_details}")

class User:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.followed_traders = set()
        self.notifications = []

    def follow_trader(self, trader_id: str):
        self.followed_traders.add(trader_id)

    def notify(self, message: str):
        self.notifications.append(message)
        print(f"Notification for user {self.user_id}: {message}")

class TradeReplicator:
    def __init__(self, trading_engine: TradingEngine):
        self.trading_engine = trading_engine
        self.users = {}
        self.trade_callbacks = []

    def register_user(self, user: User):
        self.users[user.user_id] = user

    def on_trade(self, callback: Callable[[Trade], None]):
        self.trade_callbacks.append(callback)

    def replicate_trade(self, trade: Trade):
        self.trading_engine.execute_trade(trade)
        for callback in self.trade_callbacks:
            callback(trade)

    def notify_users(self, trade: Trade):
        for user in self.users.values():
            if trade.trader_id in user.followed_traders:
                user.notify(f"Trade executed: {trade.trade_details}")

def main():
    trading_engine = TradingEngine()
    trade_replicator = TradeReplicator(trading_engine)

    # Create users
    user1 = User("user1")
    user2 = User("user2")

    # Users follow traders
    user1.follow_trader("trader1")
    user2.follow_trader("trader2")

    # Register users with the trade replicator
    trade_replicator.register_user(user1)
    trade_replicator.register_user(user2)

    # Register notification callback
    trade_replicator.on_trade(trade_replicator.notify_users)

    # Simulate trades
    trade1 = Trade("trader1", {"symbol": "AAPL", "action": "buy", "quantity": 10})
    trade2 = Trade("trader2", {"symbol": "GOOGL", "action": "sell", "quantity": 5})

    # Replicate trades
    trade_replicator.replicate_trade(trade1)
    trade_replicator.replicate_trade(trade2)

if __name__ == "__main__":
    main()
```

This code defines a system where users can follow traders and receive notifications when trades are executed. The `TradeReplicator` class handles the replication of trades and user notifications. The `TradingEngine` class simulates the execution of trades. Users can follow traders and receive notifications about their trade actions. The code is modular and testable, allowing for easy extension and maintenance.