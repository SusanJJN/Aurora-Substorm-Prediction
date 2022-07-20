from util.congfig import load_dataset_root


def test_load_dataset():
    r = load_dataset_root("radar")
    m = load_dataset_root("moving mnist")
    m2 = load_dataset_root("moving mnist++")

    print(r)
    print(m)
    print(m2)

