"""Team news: an LLM extracts structured facts; plain code turns facts into numbers.

The split is deliberate. The language model only reads text and fills a fixed form
(who is out, why, how sure). It never produces a probability or a stake. Converting facts
into a model input (expected loss of XI value) is ordinary code that can be unit-tested.
That way the LLM's job can be measured (precision/recall against labelled articles) and
a bad answer cannot silently become a bet.
"""
