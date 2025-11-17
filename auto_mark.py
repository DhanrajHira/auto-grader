import json
import csv
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial
import logging
import operator

from omr import mark_file, mark_single_file
from pathlib import Path

from rapidfuzz.distance import Levenshtein

logger = logging.getLogger("CLI")
logger.setLevel(logging.WARNING)

DISTANCE_THRESHOLD = 3


def load_student_ids(student_id_fname):
    logger.info(f"Loading student ids from {student_id_fname}")
    if not student_id_fname:
        return None
    file = Path(student_id_fname)
    with file.open() as f:
        all_ids = f.read().split("\n")
    filtered = filter(lambda s: s, all_ids)
    return list(map(lambda s: s.strip(), filtered))


def get_closest_matching_student_id(student_id, known_student_ids):
    logger.info(f"Finding closest match for {student_id}")
    min_distance = 100
    best_match = None
    for known_student_id in known_student_ids:
        distance = Levenshtein.distance(student_id, known_student_id)
        if distance < min_distance:
            min_distance = distance
            best_match = known_student_id
    logger.info(f"Found closest match as {best_match} with distance of {min_distance}")
    return best_match, min_distance


def do_fix_detected_student_ids(known_ids, student_ids):
    fixed_student_ids = []
    for student_id in student_ids:
        closest_match, distance = get_closest_matching_student_id(student_id, known_ids)
        if distance <= DISTANCE_THRESHOLD:
            fixed_student_ids.append(closest_match)
        else:
            logger.warning(
                "Found no known student id that was close enough "
                f"for {student_id}, the best match was {closest_match}"
            )
            fixed_student_ids.append(student_id)
    return fixed_student_ids


def fix_detected_student_ids(known_student_ids_fname, student_ids_fname):
    logger.info("Fixing student ids")
    student_ids = load_student_ids(student_ids_fname)
    known_ids = load_student_ids(known_student_ids_fname)
    if not student_ids:
        if known_ids:
            logger.warning(
                "Known student IDs were provided but not the detected student IDs."
            )
            return None
        else:
            logger.warning("Known and detected student IDs were not provided")
        return None

    if not known_ids:
        logger.info("Known student IDs not found, not fixing student IDs.")
        return student_ids

    return do_fix_detected_student_ids(known_ids, student_ids)


def build_output_path(attempt_file_path, output_fname_pattern):
    output_fname = output_fname_pattern.replace("%F", attempt_file_path.name)
    return attempt_file_path.parent / output_fname


def mark_and_write_graded(answer_file_path, attempt_file_path, /, output_fname_pattern):
    answer_file = Path(answer_file_path)
    with answer_file.open("rb") as answer_file:
        with attempt_file_path.open("rb") as attempt_file:
            (score, total_score), marked_file_content = mark_file(
                attempt_file.read(), answer_file.read()
            )

    output_file = build_output_path(attempt_file_path, output_fname_pattern)
    with output_file.open("wb+") as f:
        f.write(marked_file_content)
    return score, total_score


def write_scores_to_outfile(scores_pair, outfile_name):
    outfile = Path(outfile_name).resolve()
    logger.debug(f"Writing {scores_pair}")
    with outfile.open("w+") as f:
        if outfile_name.endswith(".json"):
            scores_dict = dict(scores_pair)
            json.dump(scores_dict, f, indent=4)
        else:
            writer = csv.writer(f)
            writer.writerows(scores_pair)


def do_mark_single_file(args, file_to_mark):
    answer_file = Path(args.answer_file).resolve()
    with file_to_mark.open("rb") as attempt_file:
        with answer_file.open("rb") as answer_file:
            scores, marked_file_content = mark_single_file(
                attempt_file.read(), answer_file.read()
            )

    output_file = build_output_path(file_to_mark, args.out_fname_pat)
    with output_file.open("wb+") as f:
        f.write(marked_file_content)

    student_ids = fix_detected_student_ids(
        args.known_student_ids_fname, args.student_ids_fname
    )
    if not student_ids:
        logger.warning(
            f"Skipping writing scores to {args.out_file} file since "
            "student id file could not be found."
        )
        return

    num_student_ids = len(student_ids)
    num_scores = len(scores)
    if num_student_ids != num_scores:
        logger.warning(
            f"The number of attempts is {num_scores} while "
            f"the number of student IDs is {num_student_ids}"
        )

    results = zip(student_ids, map(operator.itemgetter(0), scores))
    write_scores_to_outfile(results, args.out_file)


def file_name_to_student_id(fname):
    return fname[:-4]


def do_mark(args):
    to_mark = Path(args.to_mark).resolve()
    if to_mark.is_dir() and args.single_file:
        print(
            "Argument --single-file only makes sense when a file is being marked",
            str(to_mark),
            "is a directory",
        )
        exit(1)
    files_to_mark = list(to_mark.glob("*.pdf")) if to_mark.is_dir() else (to_mark,)
    if not args.single_file:
        if args.student_ids_fname:
            print(
                "Argument --student-ids only makes sense when marking in single file mode."
            )
            print("Otherwise, the program expects the file names to be student IDs. ")
            exit(1)

        pool = Pool(args.threads)
        marker = partial(
            mark_and_write_graded,
            args.answer_file,
            output_fname_pattern=args.out_fname_pat,
        )
        results = pool.map(marker, files_to_mark)
        results = [
            (file_name_to_student_id(str(f.name)), score)
            for f, (score, _) in zip(files_to_mark, results)
        ]
        known_student_ids = load_student_ids(args.known_student_ids_fname)
        if not known_student_ids:
            logger.warning(
                "Could not load known student ids, not fixing detected student ids."
            )
            write_scores_to_outfile(results, args.out_file)
            return

        detected_student_ids = list(map(operator.itemgetter(0), results))
        fixed_student_ids = do_fix_detected_student_ids(
            known_student_ids, detected_student_ids
        )
        results = zip(
            fixed_student_ids,
            map(operator.itemgetter(1), results),
        )
        write_scores_to_outfile(results, args.out_file)
    else:
        assert len(files_to_mark) == 1
        file_to_mark = files_to_mark[0]
        do_mark_single_file(args, file_to_mark)


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    mark_parser = subparsers.add_parser("mark")
    mark_parser.add_argument("to_mark", help="The file to mark.")
    mark_parser.add_argument("answer_file", help="The answer key file.")
    mark_parser.add_argument(
        "-j",
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use (only affects batch mode).",
    )
    mark_parser.add_argument(
        "-o",
        "--output",
        dest="out_file",
        default="scores.csv",
        help="The file to write the test scores to.",
    )
    mark_parser.add_argument(
        "--student-ids",
        dest="student_ids_fname",
        default=None,
        help="The detected student IDs in the same order as the attempts are in"
        " the to_mark file. (only meaningful when using --single-file)",
    )
    mark_parser.add_argument(
        "--known-student-ids",
        dest="known_student_ids_fname",
        default=None,
        help="Known student IDs to correct the detected ones.",
    )
    mark_parser.add_argument(
        "-p",
        "--graded-file-name",
        default="graded_%F",
        dest="out_fname_pat",
        help="The pattern that specifies the name of the output PDF file with"
        " the markings",
    )
    mark_parser.add_argument(
        "--single-file",
        default=False,
        action="store_true",
        dest="single_file",
        help="Use when all the attempts are in a single file.",
    )

    args = parser.parse_args()
    if args.command == "mark":
        do_mark(args)


if __name__ == "__main__":
    main()
