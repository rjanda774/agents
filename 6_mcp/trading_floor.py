from traders import Trader
from typing import List
import asyncio
from tracers import LogTracer
from agents import add_trace_processor
from market import is_market_open
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
        # Trader.run() and is_market_open() already catch everything they can
        # anticipate (a single trader's cycle failing, or Polygon being unreachable),
        # so neither should normally raise here. But this try/except is the very last
        # line of defense around the whole scheduler loop: without it, ANY unexpected
        # exception this iteration -- even one neither of those guards was written to
        # anticipate -- would propagate straight out of run_every_n_minutes(), and
        # since it's the coroutine asyncio.run() is driving, that kills the entire
        # process permanently with nothing left running to retry on the next tick.
        # That's the same failure mode the is_market_open() and portfolio-value-tail
        # fixes closed elsewhere; this closes the last remaining gap, at the top level.
        try:
            if RUN_EVEN_WHEN_MARKET_IS_CLOSED or is_market_open():
                await asyncio.gather(*[trader.run() for trader in traders])
            else:
                print("Market is closed, skipping run")
        except Exception as e:
            import traceback
            print(f"Unexpected error in scheduler loop: {e}")
            traceback.print_exc()
        await asyncio.sleep(RUN_EVERY_N_MINUTES * 60)


if __name__ == "__main__":
    print(f"Starting scheduler to run every {RUN_EVERY_N_MINUTES} minutes")
    asyncio.run(run_every_n_minutes())
