"""Primitives shared by the official event sources (FOMC, SEC EDGAR): canonical hashing, HTTP clock
evidence, the append-only record store with content-addressed raws, server-attested causal
availability and the rolling request limiter. Each source keeps its own spec, identity and rules."""
