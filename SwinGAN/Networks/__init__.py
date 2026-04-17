"""
Keep SwinGAN.Networks import-light.

Do not import large modules at package import time (helps CPU-only tests and avoids
pulling in optional deps unless needed).
"""

__all__ = []
