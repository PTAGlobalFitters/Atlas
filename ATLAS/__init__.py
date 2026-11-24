
import os

# Let JAX take GPU memory as it needs it instead of reserving 75% of every
# visible GPU when it starts.  setdefault keeps any value the user already set
# (e.g. XLA_PYTHON_CLIENT_PREALLOCATE=true to get preallocation back).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

# Used to enable 64-bit mode in JAX for better precision in calculations.
if not jax.config.read('jax_enable_x64'):
    # Enable 64-bit mode.  Checked through the config, not by creating an
    # array: an array would initialise the JAX backend at import time, and on
    # a GPU machine every process that imports ATLAS (e.g. the atlas-run
    # driver) would then reserve most of each visible GPU's memory.
    jax.config.update('jax_enable_x64', True)

    if jax.config.read('jax_enable_x64'):
        print("JAX 64-bit mode automatically enabled successfully.")
    else:
        print("JAX default dtype is still not float64 even after enabling. "
              "You may lose needed precision for many calculations.")
