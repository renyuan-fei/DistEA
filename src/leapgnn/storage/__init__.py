"""
Vendored from LeapGNN-AE/storage/

Only storage_dist is included; storage_naive and storage_local are not
used by DistEA and are intentionally omitted.
"""

from .storage_dist import DistCacheClient, DistCacheClientP3
