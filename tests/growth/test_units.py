import pytest

from scripts.growth.render_units import render


def test_only_new_user_units_are_rendered_to_staging(tmp_path):
    output = tmp_path / 'units'
    paths = {'growth_release': '/synthetic/growth', 'site_release': '/synthetic/site', 'site_env': '/synthetic/config/site.env', 'python': '/synthetic/venv/bin/python', 'node': '/synthetic/bin/node'}
    names = render(output, **paths)
    assert sorted(p.name for p in output.iterdir()) == sorted(names)
    referral = (output / 'hyprl-growth-referrals.service').read_text()
    assert 'ExecStart=/synthetic/venv/bin/python -m scripts.growth.discord_bot' in referral
    assert 'EnvironmentFile=%h/.config/hyprl-growth/referral.env' in referral
    assert 'MemoryMax=160M' in referral
    assert 'After=hyprl-site.service' in (output / 'hyprl-growth-newsletter.service').read_text()
    timer = (output / 'hyprl-growth-newsletter.timer').read_text()
    assert 'Mon *-*-* 18:00:00 Europe/Paris' in timer and 'Persistent=true' in timer
    assert all('@' not in p.read_text() for p in output.iterdir())
    with pytest.raises(ValueError, match='empty'):
        render(output, **paths)


@pytest.mark.parametrize('bad', ['relative', '/tmp/a b', '/tmp/a\nb', '/tmp/%h', '/tmp/"x', '/tmp/\\x'])
def test_unit_paths_cannot_inject_systemd_directives(tmp_path, bad):
    with pytest.raises(ValueError, match='paths'):
        render(tmp_path / 'units', growth_release=bad)
    assert not (tmp_path / 'units').exists()
