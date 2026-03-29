## Summary

Describe the change and the reason for it.

## Testing

- `cargo test`
- `cargo check --all-features`
- `cargo fmt --all`
- `cargo clippy --all-targets --all-features -- -D warnings`

List any deviations, skipped steps, or live-provider tests that were intentionally not run.

## Checklist

- [ ] The change is focused and does not bundle unrelated refactors.
- [ ] Tests or validation steps were added or updated when appropriate.
- [ ] Documentation was updated for public API, setup, or behavior changes.
- [ ] Secrets, private data, and generated artifacts are not included.
- [ ] Benchmark impact is noted if retrieval, ranking, or scoring behavior changed.

## Context

Link the related issue, design note, or benchmark discussion if applicable.
