import os

# Force offline demo mode before evalforge is imported: config and client are
# module-level singletons built from the environment at import time.
os.environ["MOCK_MODE"] = "true"
os.environ["PROVIDER"]  = "mock"
os.environ["MAX_CONCURRENT_CALLS"] = "3"
