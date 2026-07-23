# Contributing

All changes must be submitted through a pull request. Direct pushes to
`main` are not accepted.

Keep contributions within the existing public source boundary. Do not include
credentials, environment files, production endpoints, deployment manifests,
private datasets, ground truth, reference solutions, generated artifacts, or
tool-specific agent instructions.

Tests and examples must use synthetic data. Before opening a pull request, run:

```bash
python .github/scripts/check_public_tree.py .
poetry install --with dev
poetry run pytest
```

By submitting a pull request, you confirm that you have the right to contribute
the proposed material and agree to the contribution grant in [`LICENSE`](LICENSE).
The repository remains proprietary; submitting a contribution does not grant
the public any right to reuse its source.
