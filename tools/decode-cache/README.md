# Decode-cache routing prior

`inkling-2-mini-routing-prior.bin` seeds `SQUID_DECODE_CACHE`. It holds little-endian u32 counts, `[24 layers][128 experts]`, of how often each expert was routed while decoding. A loop layer counts once per visit, so three times per token.

It was recorded from these 7 prompts, about 512 greedy tokens each (3676 tokens in all), on the uncensored model:

1. Explain how a refrigerator works, step by step, for a curious teenager.
2. Write a short story about a lighthouse keeper who finds a message in a bottle.
3. A train leaves at 3pm going 60 mph; another leaves the same station at 4pm going 80 mph on the same track. When does the second catch the first? Show your work.
4. You are an agent with tools: web_search(query), get_weather(city, date), book_hotel(city, checkin, nights). Plan a 3-day trip to Lisbon next month. First output the tool calls you would make as JSON objects, one per line, then a short itinerary.
5. Explique en français, de façon claire et détaillée, comment fonctionne la photosynthèse, puis résume en trois points.
6. Write a bash script that finds the 10 largest files under a directory, prints their sizes in human-readable form, and handles spaces in file names. Explain each line.
7. Summarize the causes and consequences of the 2008 financial crisis in a structured way, with headings.

The benchmark prompt (an LRU cache in Python) was left out, so the measured speeds don't come from a prior fitted to it.

To build a prior from your own prompts, record each one with `SQUID_ROUTE_TRACE`. The trace is written when the process exits. Then sum the traces:

```bash
SQUID_ROUTE_TRACE=/tmp/r1.trace ./zig-out/bin/squid5090 bench /opt/models/inkling-2-mini \
  --prompt "..." --max-tokens 512 --runs 1 --mode nvfp4 --embed-host --mtp 0 --ctx 65536 --kv fp8
python3 tools/decode-cache/route_prior.py my-prior.bin /tmp/r1.trace /tmp/r2.trace ...
```

Each request starts again from the prior (`SQUID_DECODE_CACHE_ADAPT=...,reset`), and then up to 16 experts are swapped every 16 tokens toward what the request uses, so the prior matters most for short replies.
