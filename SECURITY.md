# Security Guidelines for API Keys

## Security Audit Summary

This repository has been audited for API key security vulnerabilities. **No hardcoded API keys or secrets were found** in the codebase or git history.

## Current Security Status

✅ **Safe to make public** - The repository follows security best practices:

1. **API keys are loaded from environment variables** - All sensitive credentials are retrieved using `os.getenv()` rather than being hardcoded
2. **`.env` files are properly gitignored** - The `.gitignore` file excludes `.env`, `.envrc`, and similar files
3. **No secrets in git history** - A thorough search of the git history found no leaked credentials
4. **Configuration files are secret-free** - `prov.config.yaml` and other config files contain no sensitive data

## How API Keys Are Handled

The codebase uses environment variables for all API keys:

```python
# OpenAI API Key - loaded from environment
api_key = os.getenv("OPENAI_API_KEY")

# Local LLM API Key - loaded from environment with fallback
# Note: The fallback "no-key-needed" is used for local LLM servers that don't require authentication
api_key = os.getenv("LOCAL_LLM_API_KEY", "no-key-needed")
```

## Setting Up Your Environment

Before running the code, create a `.env` file in the project root with your API keys:

```bash
# .env file (DO NOT COMMIT THIS FILE)
# Replace the placeholder values below with your actual API keys
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
LOCAL_LLM_API_KEY=your-actual-local-llm-key-here
```

**Important:** The `.env` file is already in `.gitignore` and will not be committed.

## Best Practices

When contributing to this repository:

1. **Never commit API keys** - Always use environment variables
2. **Don't add secrets to notebooks** - Clear notebook outputs before committing
3. **Use `.env` files** - Keep all secrets in `.env` files that are gitignored
4. **Review before pushing** - Check `git diff` to ensure no secrets are included

## Files That Handle API Keys

The following files reference API keys (all using secure environment variable patterns):

- `src/llm_wrapper.py` - Main LLM wrapper class
- `src/LLMQA.py` - Question answering module
- `src/LLMQAv2.py` - Question answering module v2
- `src/LLMQAv3.py` - Question answering module v3
- `utils/arguments.py` - Command-line argument handling
- `utils/generation.py` - Text generation utilities

## Reporting Security Issues

If you discover a security vulnerability, please report it responsibly by:

1. **Do not** create a public issue
2. Contact the repository maintainers directly
3. Provide details about the vulnerability and steps to reproduce
