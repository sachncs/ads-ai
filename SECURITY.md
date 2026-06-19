# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2.0 | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within ads-ai, please send an email to **chakravarthy1393966@gmail.com**. All security vulnerabilities will be promptly addressed.

**Please do not report security vulnerabilities through public GitHub issues.**

### What to Include

When reporting a vulnerability, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if available)

### Response Expectations

- **Acknowledgment**: Within 48 hours of your report
- **Initial assessment**: Within 1 week
- **Resolution timeline**: Depends on severity, typically 1-4 weeks

## Disclosure Policy

- We will confirm receipt of your report within 48 hours.
- We will provide an estimated timeline for a fix.
- We will notify you when the vulnerability has been fixed.
- We request that you do not publicly disclose the issue until we have had a chance to address it.

## Security Best Practices

When using ads-ai:

- **API Keys**: Never commit `GEMINI_API_KEY` or any secrets to version control. Use environment variables or a `.env` file (which is gitignored).
- **Dependencies**: Keep dependencies up to date. Run `pip install --upgrade` periodically.
- **Network**: The pipeline makes outbound calls to Google's GenAI API. Ensure your network allows these connections.
- **Output Files**: Generated artifacts (videos, JSON) are saved to `outputs/`. Restrict file permissions if deploying in shared environments.

## Known Security Considerations

- The pipeline executes LLM-generated content. While guardrails are in place, review generated outputs before publishing.
- Video generation via Veo requires Google AI Studio API access with appropriate quotas.
