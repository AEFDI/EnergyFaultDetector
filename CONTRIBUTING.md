# Contributing to Energy Fault Detector
Thanks for your interest in contributing!

## Getting help and reporting issues

- Bug reports and feature requests: open an issue at https://github.com/AEFDI/EnergyFaultDetector/issues
- For general questions or integration support, you can also contact [aefdi@iee.fraunhofer.de](mailto:aefdi@iee.fraunhofer.de)

Please include:
- A clear description of the problem or request
- Steps to reproduce (for bugs)
- Your Python version and operating system
- The Energy Fault Detector version (`pip show energy-fault-detector`)

---
## Development setup
1. Fork the repository on GitHub and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/EnergyFaultDetector.git
   cd EnergyFaultDetector
2. Create and activate a virtual environment (example with venv):
   ```bash
    python -m venv .venv
    source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   ```
3. Install the package in editable mode with development dependencies: `pip install -e .[dev]`
4. Run the tests to verify your setup: `pytest --cov`

## Coding guidelines
- Follow the existing style and structure in the codebase.
- Keep functions and public APIs documented with clear docstrings (we use Google Style docstrings).
- Prefer small, focused changes with clear motivation.
- If you introduce new functionality, add tests where feasible.

## Pull request process
1. If you plan a substantial change, open an issue first to discuss the idea.
2. Create a feature branch: `git checkout -b feature/my-change`
3. Make your changes and add tests or documentation updates as needed.
4. Ensure tests pass: `pytest --cov`
5. Commit with a clear message and push your branch: git push origin feature/my-change
6. Open a pull request against the main branch on GitHub and describe:
   - What problem the change solves
   - How it is implemented
   - Any breaking changes or migration steps

---
## Documentation contributions
The documentation is built with Sphinx under `docs/`.
To build the docs locally:
```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser to inspect the result.
If you move or extend conceptual content (for example “Key Concepts” or “Available Models”) from the README, please:
- Update or create the relevant `.rst` pages under `docs/`
- Add them to the toctree in `docs/index.rst` if appropriate

---

By contributing to this project, you agree that your contributions will be licensed under the project’s MIT license.
