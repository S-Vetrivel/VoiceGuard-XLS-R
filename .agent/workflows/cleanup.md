---
description: Remove the virtual environment and clean up temporary files
---

1. Deactivate venv (if active):
```bash
deactivate
```

2. Delete the venv folder:
// turbo
```bash
rm -rf venv
```

3. Remove temporary files:
// turbo
```bash
rm -rf app/__pycache__ .pytest_cache temp_*
```
