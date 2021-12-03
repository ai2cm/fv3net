import requests
import json
import dotenv
import os

dotenv.load_dotenv()

TOKEN = os.environ["SLACK_WEBHOOK_TOKEN"]


def post_message(msg):

    url = f"https://hooks.slack.com/services/{TOKEN}"

    requests.post(
        url,
        headers={"Content-type": "application/json"},
        data=json.dumps({"text": msg}),
    )
