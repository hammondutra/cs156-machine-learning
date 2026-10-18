
"""Stage 1: Data preparation for the Brawl Stars ML pipeline."""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

BLUE = ("Blue_1", "Blue_2", "Blue_3")
RED = ("Red_1", "Red_2", "Red_3")
SLOTS = BLUE + RED

REQUIRED = (
    "Match_ID", "Pair_ID", "Blue_Win", "Result",
    "Mode", "Map", "Record_Type", "Is_Flipped",
    "Trophy_Change",
) + SLOTS

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

def read_matches(path):
    """Read CSV files and convert relevant fields."""

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or set(REQUIRED) - set(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}")

        rows = list(reader)

    for row in rows:
        for col in ("Match_ID", "Pair_ID", "Blue_Win", "Is_Flipped"):
            row[col] = int(row[col])

        for col in SLOTS:
            row[col] = row[col].strip().upper()

            if not row[col]:
                raise ValueError("Empty brawler name")

        if row["Blue_Win"] not in (0, 1):
            raise ValueError(f"Invalid outcome: {row['Blue_Win']}")

        # Same brawler cannot appear twice on one team.
        # But it may appear on both opposing teams.
        if (
            len(set(row[c] for c in BLUE)) != 3
            or len(set(row[c] for c in RED)) != 3
        ):
            raise ValueError(
                f"Repeated brawler within match {row['Match_ID']}"
            )

        expected_result = (
            "victory" if row["Blue_Win"] else "defeat"
        )

        if row["Result"].strip().lower() != expected_result:
            raise ValueError(
                f"Inconsistent result: match {row['Match_ID']}"
            )

    return rows


# ------------------------------------------------------------
# 2. SIDE-SWAP AUGMENTATION
# ------------------------------------------------------------

def flip_match(row):
    """Swap Blue and Red and invert the match outcome."""

    flipped = row.copy()

    for blue, red in zip(BLUE, RED):
        flipped[blue] = row[red]
        flipped[red] = row[blue]

    flipped["Blue_Win"] = 1 - row["Blue_Win"]
    flipped["Result"] = (
        "Victory" if flipped["Blue_Win"] else "Defeat"
    )

    flipped["Match_ID"] = f"mirror-{row['Match_ID']}"
    flipped["Is_Flipped"] = 1
    flipped["Record_Type"] = "FLIPPED"
    flipped["Trophy_Change"] = ""

    return flipped


# ------------------------------------------------------------
# 3. VALIDATE THE DATASET
# ------------------------------------------------------------

def verify_dataset(originals, augmented):
    """Check that each real match has one correct mirror."""

    original_ids = [r["Pair_ID"] for r in originals]

    if len(set(original_ids)) != len(originals):
        raise ValueError("Duplicate Pair_ID in original data")

    if any(
        r["Record_Type"] != "ORIGINAL"
        or r["Is_Flipped"] != 0
        for r in originals
    ):
        raise ValueError("Raw data must contain only originals")

    if len(augmented) != 2 * len(originals):
        raise ValueError("Unexpected augmented dataset size")

    augmented_ids = [r["Match_ID"] for r in augmented]

    if len(set(augmented_ids)) != len(augmented):
        raise ValueError("Duplicate Match_ID in augmented data")

    by_pair = defaultdict(list)

    for row in augmented:
        by_pair[row["Pair_ID"]].append(row)

    if set(by_pair) != set(original_ids):
        raise ValueError("Augmented Pair_ID mismatch")

    for original in originals:
        pair_id = original["Pair_ID"]
        matches = by_pair[pair_id]

        if len(matches) != 2:
            raise ValueError(f"Invalid pair size: {pair_id}")

        observed = [
            r for r in matches
            if r["Record_Type"] == "ORIGINAL"
            and r["Is_Flipped"] == 0
        ]

        mirrored = [
            r for r in matches
            if r["Record_Type"] == "FLIPPED"
            and r["Is_Flipped"] == 1
        ]

        if len(observed) != 1 or len(mirrored) != 1:
            raise ValueError(f"Invalid pair types: {pair_id}")

        expected_mirror = flip_match(original)

        for col in (*SLOTS, "Blue_Win", "Mode", "Map"):
            if observed[0][col] != original[col]:
                raise ValueError(
                    f"Original mismatch in pair {pair_id}"
                )

        for col in (
            *SLOTS, "Blue_Win", "Mode", "Map", "Pair_ID"
        ):
            if mirrored[0][col] != expected_mirror[col]:
                raise ValueError(
                    f"Incorrect mirror for pair {pair_id}"
                )


# ------------------------------------------------------------
# 4. FIVE-FOLD STRATIFIED CROSS-VALIDATION
# ------------------------------------------------------------

def make_folds(originals, k=5, seed=42):
    """Split only real matches into k folds.

    Stratify by:
      1. Brawl Ball versus other modes
      2. Blue win versus Blue loss

    This preserves approximate class and mode proportions.
    """

    groups = defaultdict(list)

    for index, row in enumerate(originals):
        key = (
            row["Mode"] == "Brawl Ball",
            row["Blue_Win"]
        )
        groups[key].append(index)

    rng = random.Random(seed)
    test_buckets = [[] for _ in range(k)]

    for key in sorted(groups):
        indices = groups[key][:]
        rng.shuffle(indices)

        # Distribute each stratum across the five folds.
        for position, index in enumerate(indices):
            test_buckets[position % k].append(index)

    all_indices = set(range(len(originals)))
    folds = []

    for test_indices in test_buckets:
        test_ids = set(test_indices)

        train = [
            originals[i]
            for i in sorted(all_indices - test_ids)
        ]

        test = [
            originals[i]
            for i in sorted(test_ids)
        ]

        folds.append((train, test))

    # Each independent match must be tested exactly once.
    held_out_ids = [
        row["Pair_ID"]
        for _, test in folds
        for row in test
    ]

    if (
        len(held_out_ids) != len(originals)
        or len(set(held_out_ids)) != len(originals)
    ):
        raise AssertionError("Cross-validation split is invalid")

    return folds


# ------------------------------------------------------------
# 5. FEATURE ENCODING
# ------------------------------------------------------------

def encode(rows, names, style="signed"):
    """Convert brawler compositions to numerical vectors.

    signed:
        +1 if only on Blue, -1 if only on Red, 0 otherwise.

    separate:
        One indicator per brawler per team.
    """

    lookup = {name: i for i, name in enumerate(names)}

    if style not in ("signed", "separate"):
        raise ValueError("Unsupported feature representation")

    width = (
        len(names)
        if style == "signed"
        else 2 * len(names)
    )

    vectors = []

    for row in rows:
        vector = [0] * width

        for col in BLUE:
            if row[col] in lookup:
                vector[lookup[row[col]]] += 1

        for col in RED:
            if row[col] in lookup:
                i = lookup[row[col]]

                if style == "signed":
                    vector[i] -= 1
                else:
                    vector[len(names) + i] += 1

        vectors.append(vector)

    return vectors


def prepare(train, test, style="signed"):
    """Prepare X and y for one experimental condition."""

    if style == "prior":
        names = []
        x_train = [[] for _ in train]
        x_test = [[] for _ in test]

    else:
        # Learn the vocabulary from training data only.
        names = sorted({
            row[col]
            for row in train
            for col in SLOTS
        })

        x_train = encode(train, names, style)
        x_test = encode(test, names, style)

    return {
        "X_train": x_train,
        "y_train": [r["Blue_Win"] for r in train],
        "X_test": x_test,
        "y_test": [r["Blue_Win"] for r in test],
        "test_pair_ids": [r["Pair_ID"] for r in test],
        "feature_names": names,
    }


# ------------------------------------------------------------
# 6. PREPARE EXPERIMENTS E0-E4
# ------------------------------------------------------------

def experiment_sets(train, test):
    """Create training/testing data for all five experiments."""

    train_ids = {r["Pair_ID"] for r in train}
    test_ids = {r["Pair_ID"] for r in test}

    if train_ids & test_ids:
        raise AssertionError("Train/test leakage detected")

    # Augment TRAINING data only.
    mirrored_train = train + [
        flip_match(row) for row in train
    ]

    # Mode-specific experiment.
    brawl_train = [
        r for r in train if r["Mode"] == "Brawl Ball"
    ]

    brawl_test = [
        r for r in test if r["Mode"] == "Brawl Ball"
    ]

    brawl_augmented = brawl_train + [
        flip_match(row) for row in brawl_train
    ]

    return {
        "E0_prior": prepare(train, test, "prior"),

        "E1_original_signed": prepare(
            train, test, "signed"
        ),

        "E2_augmented_signed": prepare(
            mirrored_train, test, "signed"
        ),

        "E3_original_separate": prepare(
            train, test, "separate"
        ),

        "E4_brawl_only": prepare(
            brawl_augmented, brawl_test, "signed"
        ),

        "E5_brawl_original": prepare(
            brawl_train, brawl_test, "signed"
),
    }

# ------------------------------------------------------------
# 7. RUN SETUP AND INSPECT RESULTS
# ------------------------------------------------------------

def main():
    originals = read_matches(
        DATA_DIR / "raw_matches.csv"
    )

    augmented = read_matches(
        DATA_DIR / "augmented_matches.csv"
    )

    verify_dataset(originals, augmented)

    print(f"Real matches: {len(originals)}")
    print(f"Augmented rows: {len(augmented)}")

    print("Original outcomes:", dict(
        sorted(Counter(
            r["Blue_Win"] for r in originals
        ).items())
    ))

    print("Game modes:", dict(
        Counter(r["Mode"] for r in originals)
    ))

    folds = make_folds(originals, k=5, seed=42)

    for number, (train, test) in enumerate(
        folds, start=1
    ):
        experiments = experiment_sets(train, test)

        print(
            f"\nFold {number}: "
            f"{len(train)} training originals, "
            f"{len(test)} testing originals"
        )

        for name, data in experiments.items():
            print(
                f"  {name}: "
                f"{len(data['X_train'])} train, "
                f"{len(data['X_test'])} test, "
                f"{len(data['X_train'][0])} features"
            )

    print("\nAll data checks passed.")
    print("Models have NOT been trained.")


if __name__ == "__main__":
    main()