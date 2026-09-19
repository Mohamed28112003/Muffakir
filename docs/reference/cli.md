# Command line interface

The `muffakir` command starts ComposerUI and prints package version information.

## Show the installed version

```bash
muffakir --version
```

## Start ComposerUI

```bash
muffakir serve
```

Available server options:

| Option | Description |
| --- | --- |
| `--host` | Bind address. Default: `127.0.0.1`. |
| `--port` | Bind port. Default: `2811`. |
| `--reload` | Enable Uvicorn auto-reload for development. |
| `--open` | Open ComposerUI in a browser after startup. |
| `--no-open` | Do not open a browser; this is the default. |
| `--log-level` | Uvicorn logging level. |
| `--runs-dir` | Store UI run data in a chosen directory for the server session. |

```bash
muffakir serve --host 127.0.0.1 --port 2812 --open
```

See [troubleshooting](../help/troubleshooting.md) when the default port cannot be bound.
