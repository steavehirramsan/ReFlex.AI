"""ReFlex.AI quickstart — a persistent agent in a dozen lines.

Run it (offline, no GPU/API key needed):

    python examples/quickstart.py

It plants a couple of facts, then asks a follow-up question on a later turn and shows that
the answer is retrieved from durable memory — along with the integrity verdict for each turn.
"""

from __future__ import annotations

import asyncio

from reflex import Agent, ReflexConfig


async def main() -> None:
    # In-memory DB so the example leaves nothing behind; drop the override to persist to disk.
    config = ReflexConfig.load(overrides={"memory": {"db_path": ":memory:"}})

    async with Agent.from_config(config) as agent:
        for message in [
            "Remember that my project is called ReFlex and it targets AMD Instinct GPUs.",
            "My deployment region is eu-west-2.",
        ]:
            turn = await agent.turn(message)
            print(f"› {message}\n  reflex: {turn.response}")
            if turn.reflection and turn.reflection.new_facts:
                print(f"  (learned: {turn.reflection.new_facts})")

        # A later turn recalls earlier facts from durable memory.
        turn = await agent.turn("Which region am I deploying to, and what hardware do I target?")
        print(f"\n› follow-up\n  reflex: {turn.response}")
        print("  retrieved:")
        for hit in turn.retrieval.hits[:4]:
            print(f"    [{hit.tier.value} {hit.score:.2f}] {hit.content}")
        print(f"\n  integrity score: {turn.integrity.score:.2f}")
        print(f"  memory tiers: {agent.stats()}")


if __name__ == "__main__":
    asyncio.run(main())
