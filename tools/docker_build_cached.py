#!/usr/bin/env python3
import subprocess
import json

import os
import sys


def most_recent_cached_commit(image):

    image_output = json.loads(
        subprocess.check_output(
            [
                "gcloud",
                "container",
                "images",
                "list-tags",
                "--limit",
                "500",
                "--format",
                "json",
                image,
            ],
        )
    )

    # prepare list of tags
    tags = {}
    for image in image_output:
        for tag in image["tags"]:
            tags[tag] = image["digest"]

    git_rev = subprocess.check_output(["git", "rev-list", "HEAD"])
    found_tag = None

    for k, line in enumerate(git_rev.splitlines()):
        rev = line.decode().strip()
        if rev in tags:
            found_tag = rev

    return found_tag


def get_rev():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def build_cached(argv):
    cache_image, argv = argv[1], argv[2:]
    commit = most_recent_cached_commit(cache_image)

    try:
        app_creds = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    except KeyError:
        ValueError(
            "Google authentication is not configure. "
            "Please set the GOOGLE_APPLICATION_CREDENTIALS environmental variable."
        )

    subprocess.check_call(
        [
            "docker",
            "build",
            "--secret",
            f"id=gcp,src={app_creds}",
            "--build-arg",
            "BUILDKIT_INLINE_CACHE=1",
            "--build-arg",
            f"COMMIT_SHA_ARG={get_rev()}",
            "--cache-from",
            cache_image + ":" + commit,
            *argv,
        ],
        env={"DOCKER_BUILDKIT": "1", "BUILDKIT_PROGRESS": "plain"},
    )


build_cached(sys.argv)
