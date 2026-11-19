"""Put src/ and src/croco/ on sys.path so the tests run from any directory."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "croco")):
    if p not in sys.path:
        sys.path.insert(0, p)
