import os

# CPU keeps the identity suite fast to compile and is what CI runs; a GPU box
# still passes, so this is a default rather than a requirement.
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"
    # Lets tests that launch GPU-capable subprocesses (the golden run) drop
    # this default without overriding a JAX_PLATFORMS the user set themselves.
    os.environ["ATLAS_TEST_CPU_DEFAULT"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
