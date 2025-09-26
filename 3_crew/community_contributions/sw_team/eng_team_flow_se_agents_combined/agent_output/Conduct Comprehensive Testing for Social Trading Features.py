```python
# Import necessary modules
import pytest
from unittest.mock import MagicMock, patch

# Assuming the following classes and functions are part of the social trading app
# These are placeholders and should be replaced with actual implementations
class User:
    def __init__(self, username):
        self.username = username

class Trade:
    def __init__(self, user, amount):
        self.user = user
        self.amount = amount

class SocialTradingApp:
    def __init__(self):
        self.users = []
        self.trades = []

    def add_user(self, user):
        self.users.append(user)

    def add_trade(self, trade):
        self.trades.append(trade)

    def get_user_trades(self, username):
        return [trade for trade in self.trades if trade.user.username == username]

# Fixtures for setting up test data
@pytest.fixture
def user():
    return User(username="test_user")

@pytest.fixture
def trade(user):
    return Trade(user=user, amount=100)

@pytest.fixture
def app(user, trade):
    app = SocialTradingApp()
    app.add_user(user)
    app.add_trade(trade)
    return app

# Unit Tests
def test_add_user(app, user):
    assert user in app.users

def test_add_trade(app, trade):
    assert trade in app.trades

def test_get_user_trades(app, user, trade):
    trades = app.get_user_trades(user.username)
    assert trade in trades

# Integration Tests
def test_integration_add_user_and_trade(app, user, trade):
    new_user = User(username="new_user")
    new_trade = Trade(user=new_user, amount=200)
    app.add_user(new_user)
    app.add_trade(new_trade)
    assert new_user in app.users
    assert new_trade in app.trades

# User Acceptance Tests
def test_user_acceptance(app):
    # Simulate user interaction
    with patch('builtins.input', side_effect=["new_user", "200"]):
        new_user = User(username=input("Enter username: "))
        new_trade = Trade(user=new_user, amount=int(input("Enter trade amount: ")))
        app.add_user(new_user)
        app.add_trade(new_trade)
        assert new_user in app.users
        assert new_trade in app.trades

# Mocking example
def test_mocking_example():
    mock_user = MagicMock(spec=User)
    mock_user.username = "mock_user"
    assert mock_user.username == "mock_user"

# Run the tests
if __name__ == "__main__":
    pytest.main()
```

This code provides a comprehensive testing strategy using `pytest` for a social trading application. It includes unit tests, integration tests, and user acceptance tests. The code also demonstrates the use of fixtures for setting up test data and mocking for simulating user interactions. The `SocialTradingApp` class is a placeholder and should be replaced with the actual implementation of the social trading application.