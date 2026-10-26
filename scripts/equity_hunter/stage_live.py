"""Stage an exact-version hunter wrapper for operator review; never run or install it."""
import argparse
from hashlib import sha256
from pathlib import Path

SOURCE_SHA256 = '83753c28e15d376e3db51a1436e6b7ce2a7cb3bb12f62c9b84ec531ad39f8b0c'


def stage(source, output):
    raw = Path(source).read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('HUNTER_SOURCE_CHANGED_REVIEW_REQUIRED')
    text = raw.decode()
    old = 'UA = json.loads((HOME / "authorizations/news-radar-sec-fed-telegram-v1.json").read_text())["scope"]["https://www.sec.gov"]["user_agent"]'
    new = '''from authorization_extensions import sec_user_agent
SEC_AUTHORIZATION = HOME / "authorizations/news-radar-sec-fed-telegram-v1.json"
UA, SEC_METADATA = ("", {}) if SCREENS_ONLY else sec_user_agent(SEC_AUTHORIZATION, datetime.datetime.now(datetime.timezone.utc))'''
    if text.count(old) != 1:
        raise ValueError('HUNTER_SOURCE_SHAPE_CHANGED')
    text = text.replace(old, new)
    old = '''def get(u, sec=False):
    for _ in range(5):
        try:
            time.sleep(0.3 if sec else 0.34); n_req["sec" if sec else "alpaca"] += 1'''
    new = '''def get(u, sec=False):
    global UA, SEC_METADATA
    for _ in range(5):
        time.sleep(0.3 if sec else 0.34)
        if sec:
            # Outside the transport retry handler; invalid authorization stops immediately.
            UA, SEC_METADATA = sec_user_agent(SEC_AUTHORIZATION, datetime.datetime.now(datetime.timezone.utc))
        try:
            n_req["sec" if sec else "alpaca"] += 1'''
    if text.count(old) != 1 or text.count('"requests": n_req') != 3:
        raise ValueError('HUNTER_SOURCE_SHAPE_CHANGED')
    text = text.replace(old, new).replace('"requests": n_req', '"requests": n_req, "sec_authorization": SEC_METADATA')
    compile(text, '<staged-hunter>', 'exec')
    # Exclusive creation prevents accidentally overwriting the live executable.
    with Path(output).open('x') as target:
        target.write(text)
    return sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(stage(args.source, args.output))


if __name__ == '__main__':
    main()
