import datetime


def generate(bucket: str, project: str, tag: str) -> str:
    """Generate GCS URL given inputs."""
    for item in [bucket, project, tag]:
        if "/" in item:
            raise ValueError(f"Slashes not permitted in bucket, project or tag: {item}")
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"gs://{bucket}/{project}/{date}/{tag}"


def main(args):
    url = generate(args.bucket, args.project, args.tag)
    print(url)


def register_parser(subparsers):
    parser = subparsers.add_parser(
        "generate", help="Generate a URL from provided bucket, project, and tag."
    )
    parser.add_argument("bucket", help="A storage bucket.")
    parser.add_argument("project", help="The name of a project.")
    parser.add_argument("tag", help="A human-readable tag.")
    parser.set_defaults(func=main)
