# Contributing to QDSeg

Thank you for your interest in contributing to QDSeg!

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Features

Feature suggestions are welcome! Please create an issue with:
- A clear description of the feature
- Use cases and benefits
- Any relevant examples or references

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Test your changes thoroughly
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Create a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Include type hints where appropriate

### Testing

- Add tests for new features
- Ensure all tests pass before submitting a PR
- Test on multiple Python versions if possible (3.8+)

## Development Setup

```bash
# Clone the repository
git clone https://github.com/jkkwoen/qdseg.git
cd qdseg

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[all]"

# Copy environment variables template
cp .env.example .env
```

## Questions?

Feel free to open an issue for any questions or discussions.
