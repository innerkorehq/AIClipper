import argparse

from .server import serve


def main() -> None:
    ap = argparse.ArgumentParser(description="AIClipper ML sidecar")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--model", default=None, help="MLX model id (overrides request model)")
    ap.add_argument("--diarization", default=None, help="pyannote diarization model id (default: provisionary)")
    args = ap.parse_args()
    serve(host=args.host, port=args.port, forced_model=args.model, diar_model=args.diarization)


if __name__ == "__main__":
    main()