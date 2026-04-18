"""Seeded RNG root and deterministic substream utilities.

Implemented in Milestone 1. Every stochastic component in leadforge must
derive its RNG from a single seeded root so that (recipe, config, seed,
version) fully determines all outputs.
"""
