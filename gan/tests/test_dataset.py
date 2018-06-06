from gan.model.dataset import SimpleDataset, MnistDataset


def test_dataset():
    for name, cls, img_size in [('mnist', MnistDataset, [28, 28, 1]),
                                ('fashion-mnist', MnistDataset, [28, 28, 1]),
                                ('pokemon', SimpleDataset, [96, 96, 4])]:
        d = cls(name)
        if not d.exists():
            print("Download the dataset before running the test please.")
            continue

        d.load()
        gen = d.next_batch(10)

        output = next(gen)
        assert output[0] == 0
        assert output[1] == 0
        assert output[2].shape == tuple([10] + img_size)
        if len(output) == 4:
            assert output[3].shape == (10, d.y_dim)

        output = next(gen)
        assert output[0] == 0
        assert output[1] == 1
