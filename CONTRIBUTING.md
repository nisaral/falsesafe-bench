# Contributing to falsesafe-bench

Thank you for your interest in contributing! This project measures safety certification errors under learned dynamics — contributions that improve the science, code quality, or accessibility are very welcome.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork locally
3. **Set up** the development environment:
   ```bash
   # Windows
   .\scripts\bootstrap.ps1
   
   # Linux/macOS
   uv venv --python 3.13
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
4. **Verify** everything works:
   ```bash
   python -m pytest -q -p no:cacheprovider
   python scripts/run_experiment.py doctor
   ```

## How to Contribute

### Bug Reports
Open an issue with:
- What you expected vs what happened
- Steps to reproduce
- Your OS, Python version, and relevant library versions

### Feature Requests
Open an issue describing:
- The problem you're trying to solve
- Your proposed solution
- How it relates to the project's research goals

### Code Contributions

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `python -m pytest -q -p no:cacheprovider`
4. Commit with a clear message
5. Push and open a Pull Request

### Areas Where Help Is Especially Valuable

| Area | Examples | Difficulty |
|---|---|---|
| **New environments** | Quadrotor, car-like, manipulator dynamics | Medium |
| **Dynamics models** | Neural ODEs, transformers, conformal prediction | Medium-Hard |
| **Safety methods** | Control barrier functions, MPC-based shields | Hard |
| **Benchmarking** | Run experiments on new systems, compare methods | Medium |
| **Documentation** | Tutorials, Jupyter notebooks, API docs | Easy-Medium |
| **Visualization** | Improved replay viewer, 3D visualization | Medium |
| **CI/CD** | GitHub Actions, testing infrastructure | Easy-Medium |

## Code Style

- Python code follows standard PEP 8
- Use type hints where practical
- Docstrings for public functions
- Comments for non-obvious logic

## Scientific Integrity Rules

These are **non-negotiable**:

1. **Never fabricate or hardcode results.** Every number comes from a logged run.
2. **Never edit `tests/` or metric definitions** without maintainer approval.
3. **Report negative results** with the same prominence as positive ones.
4. **No silent fallbacks** — don't swap datasets, shrink grids, or reduce seeds without documenting it.
5. **All runs must be reproducible** with logged seeds, configs, and commit hashes.

See [SCOPE.md](SCOPE.md) for the full integrity contract.

## Pull Request Checklist

- [ ] Tests pass (`python -m pytest -q -p no:cacheprovider`)
- [ ] Code follows the project's style
- [ ] No changes to protected test files without approval
- [ ] New features include tests where applicable
- [ ] Documentation updated if needed
- [ ] Commit messages are clear and descriptive

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
