import datetime


def resolve_url(bucket: str, project: str, tag: str) -> str:
    """Return a GCS URL given inputs."""
    for item in [bucket, project, tag]:
        if "/" in item:
            raise ValueError(f"Slashes not permitted in bucket, project or tag: {item}")
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"gs://{bucket}/{project}/{date}/{tag}"


def main(args):
    url = resolve_url(args.bucket, args.project, args.tag)
    print(url)


def register_parser(subparsers):
    parser = subparsers.add_parser(
        "resolve-url", help="Print a URL from provided bucket, project, and tag."
    )
    parser.add_argument("bucket", help="A storage bucket.")
    parser.add_argument("project", help="The name of a project.")
    parser.add_argument("tag", help="A human-readable tag.")
    parser.set_defaults(func=main)
