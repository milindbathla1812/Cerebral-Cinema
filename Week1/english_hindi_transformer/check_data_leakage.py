from config import *


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def read_pairs(en_path, hi_path):
    en = read_lines(en_path)
    hi = read_lines(hi_path)
    assert len(en) == len(hi), f"Line mismatch: {en_path} and {hi_path}"
    return list(zip(en, hi))


def report_overlap(name_a, pairs_a, name_b, pairs_b):
    set_a = set(pairs_a)
    set_b = set(pairs_b)
    overlap = set_a.intersection(set_b)
    print(f"{name_a} vs {name_b}: {len(overlap)} exact duplicate sentence-pairs")
    if overlap:
        print("Examples:")
        for i, pair in enumerate(list(overlap)[:5], 1):
            print(f"{i}. EN: {pair[0]} | HI: {pair[1]}")


def main():
    train = read_pairs(TRAIN_EN, TRAIN_HI)
    dev = read_pairs(DEV_EN, DEV_HI)
    test = read_pairs(TEST_EN, TEST_HI)

    print("Dataset sizes:")
    print("Train:", len(train))
    print("Dev:", len(dev))
    print("Test:", len(test))
    print()

    report_overlap("train", train, "dev", dev)
    report_overlap("train", train, "test", test)
    report_overlap("dev", dev, "test", test)

    print("\nImportant: tokenizer training uses ONLY train files in train_tokenizers.py")
    print("Important: model training uses train set, validation uses dev set, final test uses test set")


if __name__ == "__main__":
    main()
