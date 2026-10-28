"""Render new growth user units to an empty staging directory; never install/start them."""
import argparse
from pathlib import Path


def render(output, **paths):
    if any(not Path(v).is_absolute() or any(c.isspace() or c in '%\\"\0' for c in v) for v in paths.values()):
        raise ValueError('unit paths must be absolute and contain no whitespace or systemd escapes')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(output.iterdir()):
        raise ValueError('unit staging directory must be empty')
    source = Path(__file__).parent / 'deploy'
    names = ['hyprl-growth-referrals.service', 'hyprl-growth-newsletter.service', 'hyprl-growth-newsletter.timer']
    rendered = {}
    for name in names:
        text = (source / (name + '.in' if name.endswith('.service') else name)).read_text()
        for key, value in paths.items():
            text = text.replace('@' + key.upper() + '@', value)
        if '@' in text:
            raise ValueError('unresolved unit placeholder')
        rendered[name] = text
    for name, text in rendered.items():
        with (output / name).open('x') as stream:
            stream.write(text)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['growth-release', 'site-release', 'site-env', 'python', 'node', 'output']:
        parser.add_argument('--' + name, required=True)
    args = vars(parser.parse_args())
    output = args.pop('output')
    try:
        names = render(output, **args)
    except ValueError as error:
        parser.error(str(error))
    print('Rendered only: ' + ', '.join(names))


if __name__ == '__main__':
    main()
