# Contributing to memlocal_core

Thanks for contributing.

This project is a Rust core library for local-first agent memory, so changes tend to affect API shape, retrieval behavior, storage semantics, and benchmark results. Keep changes focused and document behavior changes clearly.

## Before You Start

- Search existing issues and pull requests before opening a new one.
- For large API, schema, or architecture changes, open an issue first so the approach can be aligned before implementation work starts.
- Keep pull requests small and narrowly scoped. Unrelated refactors slow review.

## Development Setup

Prerequisites:

- Rust stable, 1.75 or newer
- Optional: API keys in a local `.env` file for live provider-backed tests

Common commands:

```bash
git clone https://github.com/memlocal/memlocal_core.git
cd memlocal_core

cargo build
cargo test
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo check --all-features
```

If you need provider-backed local testing, copy `.env.example` to `.env` and fill only the keys you need.

## Testing Expectations

- Add or update tests for behavior changes when practical.
- Run `cargo test` for normal changes.
- Run `cargo check --all-features` when touching optional HTTP functionality.
- Run `cargo fmt --all` and `cargo clippy --all-targets --all-features -- -D warnings` before opening a pull request.

Some integration tests use live APIs and skip automatically when required environment variables are missing. Do not require maintainers to supply secrets just to review a change.

## Code Style

- Follow the existing module layout and naming patterns.
- Prefer small, explicit changes over broad refactors.
- Avoid adding new dependencies unless they are clearly justified.
- Preserve public APIs unless the change requires a break and the PR explains the migration impact.
- Update README or inline documentation when public behavior, setup, or configuration changes.

## Pull Request Guidelines

Include the following in your pull request:

- A clear summary of what changed and why
- Linked issue or context when applicable
- Notes on testing performed
- Benchmark or retrieval impact notes for ranking, scoring, or extraction changes
- Migration notes if any public API or config behavior changed

PRs may be asked to split if they combine multiple unrelated concerns.

## Secrets and Data

- Never commit API keys, tokens, or private datasets.
- Never include real user memory data, conversation logs, or private database snapshots in issues or pull requests.
- Use sanitized fixtures whenever possible.

## Review Process

Reviews are handled on a best-effort basis. Maintainers may ask for tests, narrower scope, or documentation updates before merge.
