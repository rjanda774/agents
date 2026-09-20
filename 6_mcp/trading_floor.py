from traders import Trader
from typing import List
import asyncio
from tracers import LogTracer
from agents import add_trace_processor
from market import is_market_open
from single_instance import acquire_single_instance_lock, AlreadyRunningError
from dotenv import load_dotenv
import os

load_dotenv(override=True)

RUN_EVERY_N_MINUTES = int(os.getenv("RUN_EVERY_N_MINUTES", "60"))
RUN_EVEN_WHEN_MARKET_IS_CLOSED = (
    os.getenv("RUN_EVEN_WHEN_MARKET_IS_CLOSED", "false").strip().lower() == "true"
)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

names = ["Cathie"]
lastnames = ["Income"]
model_names = [MODEL_NAME]
short_model_names = [MODEL_NAME]


def create_traders() -> List[Trader]:
    traders = []
    for name, lastname, model_name in zip(names, lastnames, model_names):
        traders.append(Trader(name, lastname, model_name))
    return traders


async def run_every_n_minutes():
    add_trace_processor(LogTracer())
    traders = create_traders()
    while True:
        if RUN_EVEN_WHEN_MARKET_IS_CLOSED or is_market_open():
            await asyncio.gather(*[trader.run() for trader in traders])
        else:
            print("Market is closed, skipping run")
        await asyncio.sleep(RUN_EVERY_N_MINUTES * 60)


if __name__ == "__main__":
    # Guard against a second trading_floor.py accidentally running alongside
    # this one -- observed in practice to cause hard-to-diagnose Schwab token
    # flakiness (two processes racing on the same schwab_token.json) and,
    # more importantly, two independent trading cycles driving Cathie's
    # account state at once. See single_instance.py for the full story.
    try:
        acquire_single_instance_lock()
    except AlreadyRunningError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)

    print(f"Starting scheduler to run every {RUN_EVERY_N_MINUTES} minutes")
    asyncio.run(run_every_n_minutes())
