import pytest

from scripts.forward_reveal_units import units


def test_only_new_research_user_units_with_dst_aware_schedule():
    rendered = units('/synthetic/research', '/synthetic/environment', '/synthetic/python', 'a' * 40)
    assert set(rendered) == {'hyprl-research-forward-reveal.service', 'hyprl-research-forward-reveal.timer'}
    service = rendered['hyprl-research-forward-reveal.service']
    timer = rendered['hyprl-research-forward-reveal.timer']
    assert '-m scripts.forward_reveal --expected-commit ' + 'a' * 40 + ' --publish' in service
    assert 'MemoryMax=300M' in service and 'CPUQuota=50%' in service
    assert 'OnCalendar=Mon..Fri *-*-* 16:15:00 America/New_York' in timer
    assert 'Persistent=true' in timer


@pytest.mark.parametrize('path', ['relative', '/synthetic/has space', '/synthetic/%n', '/synthetic/line\nbreak'])
def test_unit_paths_cannot_inject_directives(path):
    with pytest.raises(ValueError):
        units(path, '/synthetic/environment', '/synthetic/python', 'a' * 40)
