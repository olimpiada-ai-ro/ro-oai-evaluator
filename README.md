# RO-OAI Evaluator

This repository contains the public source of the RO-OAI machine-learning
submission evaluator. It includes the scoring, parsing, sandboxing, and
validation implementation together with synthetic tests.

Production credentials, deployment manifests, private datasets, ground truth,
reference submissions, and operational configuration are intentionally not
part of this repository.

## Local development

The project requires Python 3.11 and Poetry 1.7.1.

```bash
poetry install --with dev
poetry run pytest
```

Without a private runtime configuration, the message broker is disabled and
the application can be imported and tested locally without credentials.

## Contributions

Changes are accepted through pull requests. Please read
[`CONTRIBUTING.md`](CONTRIBUTING.md) before opening one.
