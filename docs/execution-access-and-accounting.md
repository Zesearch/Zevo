# Execution access and accounting

The Docker Compose dashboard proxy and execution services use separate networks.
Only the backend joins both. Optimization workers cannot directly reach the
proxy at `http://web` and inherit its dashboard credential. Requests bearing a
worker or service credential cannot use an added UI header to gain held-out
visibility. Explicit opaque browser origins (`Origin: null`) are rejected for
mutations, including with permissive CORS settings.

This is a local trusted-host deployment, not a sandbox for hostile code. Host
administrators, access to the shared database, and host gateway routes remain
outside this boundary. Deploying untrusted tenants requires authenticated,
scoped worker access and independent filesystem/network/secret isolation.

Heartbeat usage is committed with transcript batches during execution. Repeated
message snapshots replace that message's counters; final session totals replace
interim totals. Database retries write absolute counters. If execution stops
before completion, its last committed spend remains visible to budget checks.
Provider events and batching introduce a delay, so an estimated budget is not a
hard provider billing cap. Unknown model prices retain the existing zero-price
fallback; subscription plans may differ from token estimates.

Rented compute continues accruing estimated cost after a run ends until the
provider release is confirmed and `released_at` is recorded. A cancelled run is
not proof that its infrastructure has stopped billing. Existing owned-host and
cluster accounting rules remain in effect.
