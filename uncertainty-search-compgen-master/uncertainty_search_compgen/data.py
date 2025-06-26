import string
import os
import json
import re


def format_source(source):
    if source is None:
        return None

    for char in string.punctuation:
        source = source.replace(char, " %s " % char)

    return re.sub(r"\s+", " ", source)


def check_person_name(source, target):
    """Check if example does not have exact match for PersonName.

    The filtered example could be: Who is her boss ? ### ( Yield :output (
    FindManager :recipient ( Execute :intension ( refer ( extensionConstraint (
    RecipientWithNameLike :constraint ( Constraint[Recipient] ) :name # (
    PersonName " Angelina " ) ) ) ) ) ) ).

    Args:
    source: source string.
    target: target string.

    Returns:
    False if contains strings that are not exact match for PersonName, True
    otherwise.
    """
    for prefix in ("PersonName",):
        regex = r'%s " (.+?) "' % prefix
        matches = re.findall(regex, target)
        for match in matches:
            source_rhs = format_source(match)
            if source_rhs.lower() not in source.lower():
                print("`%s` is not in `%s`. %s" % (source_rhs, source, target))
                return False
        return True


def check_year(source, target):
    """Check if example does not have exact match for year.

    The filtered example usually requires additional context, e.g. Can you create
    an event at 11 for May 4 th ? ### ( Yield :output ( CreateCommitEventWrapper
    :event ( CreatePreflightEventWrapper :constraint ( Constraint[Event] :start (
    ?= ( DateAtTimeWithDefaults :date ( MDY :day # ( Number 4 ) :month # ( Month "
    MAY " ) :year # ( Number 2019 ) ) :time ( NumberAM :number # ( Number 11 ) ) )
    ) ) ) ) ).

    Args:
    source: source string.
    target: target string.

    Returns:
    False if contains strings that are not exact match for PersonName, True
    otherwise.
    """
    for arg in ("year",):
        regex = r":%s # \( Number 20(.+?)\.?0? \)" % arg
        matches = re.findall(regex, target)
        for match in matches:
            if match not in source:
                print("`%s` is not in `%s`. %s" % (match, source, target))
                return False
    return True


def retokenize_input(input_str):
    return format_source(input_str)


def filter_example(pair):
    return check_year(*pair) and check_person_name(*pair)


def parse_json_objects(lines):
    return map(json.loads, lines)


def load_smcalflow_cs(smcalflow_cs_data_dir):
    data_directory = smcalflow_cs_data_dir

    with open(os.path.join(data_directory, "train.jsonl")) as f:
        train_objs = [
            {**o, "split": "train"} for o in parse_json_objects(f.readlines())
        ]

    with open(os.path.join(data_directory, "valid.jsonl")) as f:
        val_objs = [{**o, "split": "train"} for o in parse_json_objects(f.readlines())]

    with open(os.path.join(data_directory, "test.jsonl")) as f:
        test_objs = [{**o, "split": "test"} for o in parse_json_objects(f.readlines())]

    return (
        (
            list(
                zip(
                    map(lambda x: retokenize_input(x["utterance"]), train_objs),
                    map(lambda x: x["plan"], train_objs),
                )
            )
        ),
        (
            list(
                zip(
                    map(lambda x: retokenize_input(x["utterance"]), val_objs),
                    map(lambda x: x["plan"], val_objs),
                )
            )
        ),
        (
            list(
                zip(
                    map(lambda x: retokenize_input(x["utterance"]), test_objs),
                    map(lambda x: x["plan"], test_objs),
                )
            )
        ),
    )


# +
# def load_smcalflow_cs_simplified(smcalflow_cs_data_dir):
#     data_directory = smcalflow_cs_data_dir

#     with open(os.path.join(data_directory, "train.simplified.jsonl")) as f:
#         train_objs = [
#             {**o, "split": "train"} for o in parse_json_objects(f.readlines())
#         ]

#     with open(os.path.join(data_directory, "fewshots.simplified.jsonl")) as f:
#         fs_objs = [{**o, "split": "train"} for o in parse_json_objects(f.readlines())]

#     with open(os.path.join(data_directory, "valid.simplified.jsonl")) as f:
#         val_objs = [{**o, "split": "valid"} for o in parse_json_objects(f.readlines())]

#     with open(os.path.join(data_directory, "test.simplified.jsonl")) as f:
#         test_objs = [{**o, "split": "test"} for o in parse_json_objects(f.readlines())]

#     return list(
#         map(
#             lambda x: list(filter(lambda y: y[1] is not None, x)),
#             (
#                 (
#                     list(
#                         zip(
#                             map(lambda x: retokenize_input(x["source"]), train_objs),
#                             map(lambda x: retokenize_input(x["target"]), train_objs),
#                         )
#                     )
#                 )
# + (
#                     list(
#                         zip(
#                             map(lambda x: retokenize_input(x["source"]), fs_objs),
#                             map(lambda x: retokenize_input(x["target"]), fs_objs),
#                         )
#                     )
#                 ),
#                 (
#                     list(
#                         zip(
#                             map(lambda x: retokenize_input(x["source"]), val_objs),
#                             map(lambda x: retokenize_input(x["target"]), val_objs),
#                         )
#                     )
#                 ),
#                 (
#                     list(
#                         zip(
#                             map(lambda x: retokenize_input(x["source"]), test_objs),
#                             map(lambda x: retokenize_input(x["target"]), test_objs),
#                         )
#                     )
#                 ),
#             ),
#         )
#     )

# +
import os
import json
import random
from sklearn.model_selection import train_test_split

def load_smcalflow_cs_simplified(smcalflow_cs_data_dir, train_ratio=0.9):
    data_directory = smcalflow_cs_data_dir


    def load_jsonl(file_path, split_label):
        with open(file_path) as f:
            return [{**o, "split": split_label} for o in parse_json_objects(f.readlines())]


    train_objs = load_jsonl(os.path.join(data_directory, "train.simplified.jsonl"), "train")
    fs_objs = load_jsonl(os.path.join(data_directory, "fewshots.simplified.jsonl"), "train")
    val_objs = load_jsonl(os.path.join(data_directory, "valid.simplified.jsonl"), "valid")
    test_objs = load_jsonl(os.path.join(data_directory, "test.simplified.jsonl"), "test")


    def process_data(data):
        return [
            (retokenize_input(x["source"]), retokenize_input(x["target"]), x["qid"])
            for x in data if x["target"] is not None
        ]


    train_full = process_data(train_objs) + process_data(fs_objs)


    train_large, _ = train_test_split(train_full, test_size=1-train_ratio, random_state=42)

 
    val_data = process_data(val_objs)
    test_data = process_data(test_objs)

    return train_large, train_full, val_data, test_data

# -




