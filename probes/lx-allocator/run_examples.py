"""Run every block in lx_examples.py on its own, each after `from lx_design import *`.

    python3 run_examples.py            # all blocks
    python3 run_examples.py Restriction  # blocks whose title contains the word
"""

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
BLOCKS = re.split(r"^# example: (.+)$", (HERE / "lx_examples.py").read_text(), flags=re.M)[1:]

for title, body in zip(BLOCKS[::2], BLOCKS[1::2]):
    if len(sys.argv) > 1 and not any(w in title for w in sys.argv[1:]):
        continue
    print(f"== {title}")
    namespace: dict = {}
    exec("from lx_design import *", namespace)
    exec(compile(body, f"lx_examples.py <{title}>", "exec"), namespace)
    print()
