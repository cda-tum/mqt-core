from src.mqt.core import compare


def test_zero_point_one() -> None:
    compare("example_results.json", factor=0.1, only_changed=False, sort='ratio', no_split=False)


def test_zero_point_three() -> None:
    compare("example_results.json", factor=0.3, only_changed=False, sort='ratio', no_split=False)


def test_only_changed() -> None:
    compare("example_results.json", factor=0.2, only_changed=True, sort='ratio', no_split=False)


def test_only_changed_and_no_split() -> None:
    compare("example_results.json", factor=0.2, only_changed=True, sort='ratio', no_split=True)


def test_sort_by_experiment() -> None:
    compare("example_results.json", factor=0.2, only_changed=True, sort='experiment', no_split=True)
