# Contributing

Report bugs or suggest improvements through
[GitHub Issues](https://github.com/mint-philosophy/NoRA-benchmark/issues).
Include the command, package version, and a small reproducible example.
Remove API keys and private model responses from any shared logs.

To test a code change:

```bash
uv sync --locked --extra test
make check
```

Tests make no paid API calls. Include regression tests for changes to prediction
parsing or scoring, and update the relevant usage examples when changing the CLI
or Python interface.
