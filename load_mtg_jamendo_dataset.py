from pathlib import Path
from functools import partial
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import csv

CATEGORIES = ["genre", "instrument", "mood/theme"]
TAG_HYPHEN = "---"


def load_mmap(path, dimensions):
    fp = np.memmap(path, dtype="float16", mode="r")
    embedding = np.array(fp).reshape(-1, dimensions)
    embedding = embedding.mean(axis=0)
    del fp

    return embedding


def load_mtg_jamendo_dataset(
    basedir, dimensions, merge_validation=True,
):
    basedir = Path(basedir)
    gt = {
        "train": read_file("data/autotagging_moodtheme-train.tsv")[0],
        "test": read_file("data/autotagging_moodtheme-test.tsv")[0],
        "validation": read_file("data/autotagging_moodtheme-validation.tsv")[0],
    }

    splits = ["train", "test", "validation"]

    if merge_validation:
        gt["train"].update(gt["validation"])
        splits.pop(-1)

    mlb = MultiLabelBinarizer()

    mlb.fit([v["tags"] for v in gt["train"].values()])

    labels = dict()
    data = dict()
    for split in splits:
        labels[split] = mlb.transform([v["tags"] for v in gt[split].values()])

        relative_paths = [v["path"] for v in gt[split].values()]
        paths = list(map(lambda x: (basedir / x).with_suffix(".dat"), relative_paths))
        data[split] = np.vstack(
            list(map(partial(load_mmap, dimensions=dimensions), paths))
        )

    return data, labels


def get_id(value):
    return int(value.split("_")[1])


def get_length(values):
    return len(str(max(values)))


def read_file(tsv_file):
    tracks = {}
    tags = {category: {} for category in CATEGORIES}

    # For statistics
    artist_ids = set()
    albums_ids = set()

    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            track_id = get_id(row[0])
            tracks[track_id] = {
                "artist_id": get_id(row[1]),
                "album_id": get_id(row[2]),
                "path": row[3],
                "duration": float(row[4]),
                "tags": row[5:],  # raw tags, not sure if will be used
            }
            tracks[track_id].update({category: set() for category in CATEGORIES})

            artist_ids.add(get_id(row[1]))
            albums_ids.add(get_id(row[2]))

            for tag_str in row[5:]:
                category, tag = tag_str.split(TAG_HYPHEN)

                if tag not in tags[category]:
                    tags[category][tag] = set()

                tags[category][tag].add(track_id)

                tracks[track_id][category].add(tag)

    print(
        "Reading: {} tracks, {} albums, {} artists".format(
            len(tracks), len(albums_ids), len(artist_ids)
        )
    )

    extra = {
        "track_id_length": get_length(tracks.keys()),
        "artist_id_length": get_length(artist_ids),
        "album_id_length": get_length(albums_ids),
    }
    return tracks, tags, extra
