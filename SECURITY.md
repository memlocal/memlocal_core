# Security Policy

## Supported Versions

This project is still early and does not maintain long-term support branches yet.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a Vulnerability

Do not report security vulnerabilities in public GitHub issues.

Use GitHub private vulnerability reporting if it is enabled for the repository. If private reporting is not available, contact the maintainers through a private channel before disclosing details publicly.

Please include:

- A clear description of the issue
- Affected version, commit, or file paths
- Reproduction steps or proof of concept
- Security impact and any assumptions
- Suggested mitigation if you already have one

## Response Expectations

- Initial acknowledgement: best effort within 5 business days
- Triage: reproduction and impact assessment after acknowledgement
- Disclosure: coordinated disclosure after a fix or mitigation is available

## Handling Sensitive Material

- Do not include API keys, tokens, or credentials in reports.
- Do not attach database files or logs containing private user memory data unless they have been fully sanitized.
- If a vulnerability depends on provider credentials or private data, describe the setup without sharing the secret values themselves.
