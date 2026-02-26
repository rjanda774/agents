"""
Options position tracking for credit spreads
Separate from stock holdings to avoid mixing options and stocks
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Literal


class OptionLeg(BaseModel):
    """Single option leg (put or call)"""
    symbol: str
    strike: float
    option_type: Literal["put", "call"]
    action: Literal["sell", "buy"]  # sell to open, buy to open
    contracts: int
    premium_per_contract: float  # Premium for THIS leg only
    

class CreditSpread(BaseModel):
    """A credit spread position (bull put or bear call)"""
    position_id: str  # Unique ID for this spread
    symbol: str
    spread_type: Literal["bull_put", "bear_call"]
    short_leg: OptionLeg  # Option we SOLD (collect premium)
    long_leg: OptionLeg   # Option we BOUGHT (protection)
    expiration_date: str  # YYYY-MM-DD
    opened_at: str  # Timestamp when opened
    net_premium_collected: float  # Total premium collected (short - long)
    max_loss: float  # Maximum possible loss
    status: Literal["open", "closed", "expired"]
    closed_at: str | None = None
    closing_premium_paid: float | None = None  # If closed early
    rationale: str = ""
    
    def days_to_expiration(self) -> int:
        """Calculate days remaining until expiration"""
        exp_date = datetime.strptime(self.expiration_date, "%Y-%m-%d")
        today = datetime.now()
        return (exp_date - today).days
    
    def profit_loss(self) -> float:
        """Calculate current P/L"""
        if self.status == "closed" and self.closing_premium_paid is not None:
            # Closed position: premium collected - premium paid to close
            return self.net_premium_collected - self.closing_premium_paid
        elif self.status == "expired":
            # Expired worthless: keep all premium
            return self.net_premium_collected
        else:
            # Open position: unrealized P/L (would need current option prices)
            # For now, assume we keep the premium if not closed
            return self.net_premium_collected
    
    def __repr__(self):
        return f"{self.spread_type.upper()} {self.symbol} {self.short_leg.strike}/{self.long_leg.strike} (${self.net_premium_collected:.2f} premium)"


class OptionsAccount(BaseModel):
    """Options-specific account tracking (separate from stock account)"""
    name: str
    open_positions: list[CreditSpread] = []
    closed_positions: list[CreditSpread] = []
    total_premium_collected: float = 0.0
    total_realized_pnl: float = 0.0
    
    def open_spread(self, spread: CreditSpread):
        """Open a new credit spread position"""
        self.open_positions.append(spread)
        self.total_premium_collected += spread.net_premium_collected
    
    def close_spread(self, position_id: str, closing_premium: float):
        """Close an existing spread position"""
        for i, pos in enumerate(self.open_positions):
            if pos.position_id == position_id:
                pos.status = "closed"
                pos.closed_at = datetime.now().isoformat()
                pos.closing_premium_paid = closing_premium
                
                pnl = pos.profit_loss()
                self.total_realized_pnl += pnl
                
                self.closed_positions.append(pos)
                self.open_positions.pop(i)
                return pos
        return None
    
    def total_open_risk(self) -> float:
        """Total max loss across all open positions"""
        return sum(pos.max_loss for pos in self.open_positions)
    
    def total_unrealized_pnl(self) -> float:
        """Unrealized P/L on open positions (premium collected so far)"""
        return sum(pos.net_premium_collected for pos in self.open_positions)
    
    def summary(self) -> dict:
        """Summary statistics"""
        return {
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed_positions),
            "total_premium_collected": self.total_premium_collected,
            "total_realized_pnl": self.total_realized_pnl,
            "total_unrealized_pnl": self.total_unrealized_pnl(),
            "total_open_risk": self.total_open_risk(),
        }
