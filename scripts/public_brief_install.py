"""Stage the existing brief's presentation patch; no data, model or delivery changes."""
import argparse
from pathlib import Path


def patched_source(source, root):
    """Fail closed on an unexpected brief version; the operator keeps the backup."""
    marker = '# HyprL public presentation names'
    if marker in source:
        raise ValueError('BRIEF_ALREADY_PATCHED')
    anchor = 'from pathlib import Path\n'
    disclosure = 'la rédaction est assistée par Claude, un modèle d\'IA d\'Anthropic ; '
    method = "L += ['### Méthode et limites',"
    if source.count(anchor) != 1 or source.count(disclosure) != 1 or source.count(method) != 1:
        raise ValueError('BRIEF_PRESENTATION_VERSION_UNKNOWN')
    bootstrap = (anchor + '\n' + marker + '\n' + f'sys.path.insert(0, {str(root)!r})\n'
                 'from scripts.public_agent_names import PUBLIC_METHOD, public_agent_label\n')
    result = source.replace(anchor, bootstrap)
    result = result.replace(disclosure, 'la rédaction est assistée par "\n'
                            '              f"{public_agent_label(\'opus\')}, construit sur un modèle de langage tiers d\'Anthropic ; ')
    result = result.replace(method, method + "\n          '- ' + PUBLIC_METHOD + '.',")
    # The short member edition includes the method too, before truncation.
    result = result.replace("L += ['', '> ' + DISCLOSURE,", "L += ['', '> ' + PUBLIC_METHOD + '.', '> ' + DISCLOSURE,")
    compile(result, '<patched-brief>', 'exec')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--staged', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.staged.exists() or args.source.resolve() == args.staged.resolve():
        parser.error('Stage into a new file, then test, back up and atomically install it')
    source = patched_source(args.source.read_text(), Path(__file__).resolve().parents[1])
    args.staged.write_text(source)
    args.staged.chmod(args.source.stat().st_mode & 0o777)


if __name__ == '__main__':
    main()
