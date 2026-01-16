# CI/CD Pipeline Setup Guide

## Overview

Your project has a comprehensive CI/CD pipeline configured with:
- **Pre-commit hooks** for local code quality checks
- **GitHub Actions** for automated testing and validation on push/PR
- **Multi-version testing** (Python 3.11, 3.12, 3.13)
- **Security scanning** with Bandit and Safety
- **Type checking** with mypy
- **Code coverage** tracking with pytest-cov

## Pre-Commit Hooks

### Installation

Pre-commit hooks are already installed and will run automatically before each commit.

```bash
# Activate virtual environment (required)
source .venv/bin/activate

# The hooks are already installed, but if needed:
pre-commit install
```

### Available Hooks

The following quality checks run automatically before commits:

| Hook | Purpose | Tools |
|------|---------|-------|
| **Formatting** | Code style consistency | Black, isort |
| **Linting** | Code quality issues | Ruff, Flake8 |
| **Security** | Security vulnerabilities | Bandit |
| **Type Checking** | Type safety | mypy |
| **File Validation** | File integrity | YAML, JSON, TOML checks |

### Running Manually

```bash
# Run all hooks on changed files
pre-commit run

# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

### Configuration

Pre-commit configuration is defined in `.pre-commit-config.yaml`:
- **Black**: Code formatter (line-length: 150)
- **isort**: Import sorter (black-compatible)
- **Ruff**: Fast linter with auto-fixes
- **Flake8**: Additional linting checks
- **Bandit**: Security vulnerability scanner
- **mypy**: Static type checker

## GitHub Actions Workflow

### Workflow File

The CI pipeline is defined in `.github/workflows/ci.yml`

### Jobs

#### 1. **Lint & Format** (ubuntu-latest, Python 3.11)
- Checks code formatting with Black
- Validates import sorting with isort
- Runs Ruff and Flake8 linting

#### 2. **Security Scan** (ubuntu-latest, Python 3.11)
- Scans code with Bandit for security issues
- Checks dependencies with Safety
- Non-blocking (failures don't block merge)

#### 3. **Type Check** (ubuntu-latest, Python 3.11)
- Validates type hints with mypy
- Non-blocking (failures don't block merge)

#### 4. **Test** (ubuntu-latest, multi-version matrix)
- Runs pytest on Python 3.11, 3.12, 3.13
- Generates coverage reports
- Uploads coverage to Codecov (Python 3.11 only)
- Fail-fast disabled (all versions complete)

#### 5. **Build Check** (ubuntu-latest, Python 3.11)
- Verifies dependencies install correctly
- Validates main module imports
- Generates build summary
- Depends on: Lint + Test jobs passing

### Workflow Triggers

The pipeline runs on:
- **Push** to `main` branch
- **Pull requests** targeting `main` branch

### Caching

GitHub Actions caches pip dependencies per Python version for faster builds:
- Caching key: `requirements.txt` hash + Python version
- Separate caches for each job type

## Local Development Workflow

### 1. Before Committing

```bash
# Activate virtual environment
source .venv/bin/activate

# Run pre-commit checks (happens automatically but can verify)
pre-commit run --all-files

# Run tests locally
pytest tests/ -v --cov=src

# Run type checking
mypy src/
```

### 2. Committing

```bash
# Stage changes
git add .

# Commit (pre-commit hooks run automatically)
git commit -m "Your commit message"
```

If hooks fail:
- They will auto-fix formatting issues
- Re-stage the auto-fixed files
- Commit again

### 3. Pushing & Pull Requests

```bash
# Push to your branch
git push origin your-branch

# Open pull request on GitHub
# GitHub Actions will automatically run the full CI pipeline
```

## Quality Gates

### Required Checks (Block Merge)
- ✅ **Lint & Format**: All style checks must pass
- ✅ **Tests**: All tests must pass on all Python versions
- ✅ **Build Check**: Main module must import successfully

### Optional Checks (Don't Block Merge)
- ⚠️ **Security Scan**: Fails are reported but don't block
- ⚠️ **Type Check**: Failures are reported but don't block

## Environment Variables

The project uses environment variables configured via `.env.example`:
- Copy `.env.example` to `.env` for local development
- `.env` files are gitignored for security

## Python Version Support

The project targets:
- **Minimum**: Python 3.11
- **Tested**: Python 3.11, 3.12, 3.13
- **Target versions in tools**: py311, py312, py313

## Dependencies

### Main Dependencies
See `requirements.txt` for production dependencies

### Development Dependencies
```bash
pip install -r requirements-dev.txt
```

Includes:
- pre-commit: Hook framework
- black, isort, ruff, flake8: Formatting & linting
- mypy: Type checking
- bandit, safety: Security scanning
- pytest, pytest-cov, pytest-xdist: Testing
- coverage: Coverage reporting

## Troubleshooting

### Pre-commit hooks not running
```bash
# Reinstall hooks
pre-commit install

# Verify installation
ls -la .git/hooks/pre-commit
```

### Black version issues
If you see "py313 not supported" error:
```bash
# Update pre-commit
pre-commit autoupdate

# Update Black locally
pip install --upgrade black
```

### Failed tests locally but passing in CI
```bash
# Ensure you're using the correct Python version
python --version

# Recreate CI environment
python -m venv test-env
source test-env/bin/activate
pip install -r requirements.txt
pytest tests/
```

### Coverage reports not uploading
- Codecov token is handled by GitHub Actions
- Failures to upload don't block the build
- Check the workflow log for details

## Maintenance

### Updating Tool Versions

Tools are defined in:
- `.pre-commit-config.yaml`: Pre-commit hook versions
- `requirements-dev.txt`: Development tool versions
- `.github/workflows/ci.yml`: CI tool versions

To update:
```bash
# Update pre-commit hooks
pre-commit autoupdate

# Update dev requirements
pip install --upgrade -r requirements-dev.txt
# Then update requirements-dev.txt with new versions
```

### Adding New Quality Checks

1. **Add to `.pre-commit-config.yaml`** for local checks
2. **Add to `.github/workflows/ci.yml`** for CI pipeline
3. **Document in** `requirements-dev.txt` if needed
4. **Test locally** with `pre-commit run --all-files`

## Best Practices

1. **Always activate venv before committing**
   ```bash
   source .venv/bin/activate
   ```

2. **Run pre-commit before git commit**
   - It's automatic but verify with `pre-commit run`

3. **Keep tests passing on all Python versions**
   - Test locally on 3.11, 3.12, 3.13 before pushing

4. **Fix code quality issues promptly**
   - Linting and formatting issues should be addressed
   - Don't merge PRs with lint failures

5. **Review security scan results**
   - Even though they're non-blocking, investigate findings
   - Address high-priority security issues

## Resources

- [Pre-commit documentation](https://pre-commit.com/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [pytest documentation](https://docs.pytest.org/)
- [Black code formatter](https://black.readthedocs.io/)
- [mypy type checker](https://mypy.readthedocs.io/)
- [Bandit security scanner](https://bandit.readthedocs.io/)
