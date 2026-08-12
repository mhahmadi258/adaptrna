"""Run the HTTP service.

    python -m adaptrna_agentic.cli.serve                    # 127.0.0.1:8000
    python -m adaptrna_agentic.cli.serve --warmup           # preload the backbone
    ADAPTRNA_API_TOKEN=... python -m adaptrna_agentic.cli.serve --host 0.0.0.0

This service can spend GPU hours and write code into the repository, so it binds to
loopback by default and **refuses to start** on any other address without a token. That
refusal is the point: the dangerous configuration should not be reachable by accident.
"""

from typing import Optional
import argparse
import os
import sys

from adaptrna_agentic.api.deps import TOKEN_VAR, is_loopback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adaptrna_agentic.cli.serve",
        description="Serve the AdaptRNA HTTP API.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default=None, help="ToolHub state directory")
    parser.add_argument("--warmup", action="store_true",
                        help="Load the backbone at startup instead of on first use")
    parser.add_argument("--reload", action="store_true", help="Auto-reload (development)")

    return parser


def check_binding(host: str, token: Optional[str]) -> Optional[str]:
    """The refusal message for an unsafe binding, or None if it is fine."""
    if is_loopback(host) or token:
        return None

    return (
        f"Refusing to bind to '{host}' without a token: this API can start training runs "
        f"and write code into the repository. Set {TOKEN_VAR} to a secret value, or bind "
        f"to 127.0.0.1 (the default)."
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get(TOKEN_VAR)

    problem = check_binding(args.host, token)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    import uvicorn

    from adaptrna_agentic.api.app import create_app

    app = create_app(data_dir=args.data_dir, token=token)

    if args.warmup:
        print("Loading backbone …", flush=True)
        for skipped in app.state.services.runtime.warmup():
            print(f"  skipped: {skipped}", file=sys.stderr)

    scheme_note = "token required" if token else "no token (loopback only)"
    print(f"AdaptRNA API on http://{args.host}:{args.port}  [{scheme_note}]")
    print(f"  docs:     http://{args.host}:{args.port}/docs")
    print(f"  sessions: shared with the terminal chat at {app.state.services.db_path}")

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
