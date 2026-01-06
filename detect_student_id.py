import pymupdf
import numpy as np
import cv2 as cv
import onnxruntime as ort
from rapidfuzz.distance import Levenshtein

from transformed_image import TransformedImage, show_image
from omr import detect_orientation_lines, get_image_from_page

from argparse import ArgumentParser
from pathlib import Path
import shutil


STUDENT_ID_DIGITS = 7
DISTANCE_THRESHOLD = 3


def mnist_preprocess_digit(digit):
    # 1. Center Crop (Remove excess background)
    # We find the bounding box of the digit to ensure we center the digit,
    # not just the image.
    contours, _ = cv.findContours(digit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assuming it's the digit)
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        digit = digit[y : y + h, x : x + w]

    # 2. Resize with Aspect Ratio Preservation
    # MNIST digits usually fit in a 20x20 box inside the 28x28 image.
    rows, cols = digit.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))

    digit_resized = cv.resize(digit, (cols, rows))

    # 3. Pad to 28x28
    # Create a black 28x28 canvas
    final_image = np.zeros((28, 28), dtype=np.uint8)

    # Calculate coordinates to center the resized digit
    pad_top = (28 - rows) // 2
    pad_left = (28 - cols) // 2

    # Paste the digit into the center
    final_image[pad_top : pad_top + rows, pad_left : pad_left + cols] = digit_resized
    return final_image


def detect_student_id_digits(grid: TransformedImage):
    cell_width = int(grid.width / STUDENT_ID_DIGITS)

    def bounding_rect_area(cnt):
        _, _, w, h = cv.boundingRect(cnt)
        return w * h

    digits = []
    for i in range(STUDENT_ID_DIGITS):
        right_edge = (i + 1) * cell_width
        left_edge = i * cell_width
        cell = grid.subimage(left=left_edge, right=right_edge)
        cell_contours, _ = cell.contours(cv.RETR_EXTERNAL)
        border_contour = max(cell_contours, key=bounding_rect_area)
        cv.drawContours(cell.img, [border_contour], -1, (0, 0, 0), thickness=10)
        digits.append(cell)
    return digits


def detect_student_id_grid(header: TransformedImage):
    header = header.grayscale().otsu_threshold()
    contours, _ = header.contours(cv.RETR_EXTERNAL)
    grid_contour = max(contours, key=cv.contourArea)
    if cv.contourArea(grid_contour) < 60 * 60 * 7:
        return None
    x, y, w, h = cv.boundingRect(grid_contour)
    grid = header.subimage(left=x, right=x + w, top=y, bottom=y + h)
    return grid


def get_header(page_img):
    lines = detect_orientation_lines(page_img)
    angles = [line.angle for line in lines]
    median_angle = float(np.median(angles))
    top_line = min(lines, key=lambda line: line.start_y)
    top = max(top_line.start_y, top_line.end_y)
    header = page_img.rotate(median_angle).crop_bottom(top)
    return header


def extract_student_id(page_img, ort_session):
    header = get_header(page_img)
    grid = detect_student_id_grid(header)
    if not grid:
        return None
    digits = detect_student_id_digits(grid)
    # for d in digits:
    #     show_image(d.img)
    #     preprocessed_digit = mnist_preprocess_digit(d.img)
    #     show_image(preprocessed_digit)
    batch_data = list(
        map(lambda d: (mnist_preprocess_digit(d.img) / 255).astype(np.float32), digits)
    )

    input_name = ort_session.get_inputs()[0].name
    predictions = []

    for digit in batch_data:
        current_img = digit.reshape(1, 1, 28, 28)
        outputs = ort_session.run(None, {input_name: current_img})
        pred = np.argmax(outputs[0])
        predictions.append(str(pred))
    student_id = "".join(predictions)
    return student_id


def get_closest_matching_student_id(student_id, known_student_ids):
    min_distance = 100
    best_match = None
    for known_student_id in known_student_ids:
        distance = Levenshtein.distance(student_id, known_student_id)
        if distance < min_distance:
            min_distance = distance
            best_match = known_student_id
    return best_match, min_distance


def do_fix_detected_student_id(known_ids, student_id):
    closest_match, distance = get_closest_matching_student_id(student_id, known_ids)
    if distance <= DISTANCE_THRESHOLD:
        return closest_match
    else:
        print(
            "Found no known student id that was close enough "
            f"for {student_id}, the best match was {closest_match}"
        )
        return student_id


def do_fix_detected_student_ids(known_ids, student_ids):
    fixed_student_ids = []
    for student_id in student_ids:
        fixed_student_ids.append(do_fix_detected_student_id(known_ids, student_id))
    return fixed_student_ids


def load_student_ids(student_id_fname):
    if not student_id_fname:
        return None
    print(f"Loading student ids from {student_id_fname}")
    file = Path(student_id_fname)
    with file.open() as f:
        all_ids = f.read().split("\n")
    filtered = filter(lambda s: s, all_ids)
    return list(map(lambda s: s.strip(), filtered))


def fix_detected_student_ids(known_student_ids_fname, student_ids):
    known_ids = load_student_ids(known_student_ids_fname)
    if not known_ids:
        print("Known student IDs not found, not fixing student ID(s).")
        return student_ids
    return do_fix_detected_student_ids(known_ids, student_ids)


def do_list(args):
    out_file = args.out_file or f"{args.file[:-4]}.ids.txt"
    document = pymupdf.Document(args.file)
    all_pages = list(document.pages())
    session = ort.InferenceSession("mnist-8.onnx")
    all_ids = []
    start_at = args.pages_per_attempt if args.first_as_key else 0
    for page_number in range(start_at, len(all_pages), args.pages_per_attempt):
        page = all_pages[page_number]
        page_img = get_image_from_page(page)
        student_id = extract_student_id(page_img, session) or "0000000"
        all_ids.append(student_id)
    all_ids = fix_detected_student_ids(args.known_student_ids_fname, all_ids)

    out_file = Path(out_file)
    with out_file.open("w") as f:
        f.writelines(map(lambda s: f"{s}\n", all_ids))


def get_split_out_dir(args):
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(exist_ok=True)
        return out_dir
    out_dir = Path(f"{args.file}.split").resolve()
    out_dir.mkdir(exist_ok=True)
    return out_dir


def do_split(args):
    document = pymupdf.Document(args.file)
    all_pages = list(document.pages())
    session = ort.InferenceSession("mnist-8.onnx")
    known_ids = load_student_ids(args.known_student_ids_fname)
    out_dir = get_split_out_dir(args)

    if args.first_as_key:
        answer_key = pymupdf.open()
        answer_key.insert_pdf(document, from_page=0, to_page=args.pages_per_attempt - 1)
        answer_key.save(out_dir / "answer_key.pdf")
        answer_key.close()

    start_at = args.pages_per_attempt if args.first_as_key else 0
    for page_number in range(start_at, len(all_pages), args.pages_per_attempt):
        page = all_pages[page_number]
        page_img = get_image_from_page(page)
        student_id = extract_student_id(page_img, session) or "0000000"
        if known_ids:
            fixed_student_id = do_fix_detected_student_id(known_ids, student_id)
            print(f"Detected ID as {student_id}, fixed as {fixed_student_id}")
            out_file_name = f"{fixed_student_id}.pdf"
        else:
            print(f"Detected ID as {student_id}")
            out_file_name = f"{student_id}.pdf"

        new_doc = pymupdf.open()
        new_doc.insert_pdf(document, from_page=page_number, to_page=page_number)
        new_doc.save(out_dir / out_file_name)
        new_doc.close()


def do_label(args):
    document = pymupdf.Document(args.file)
    page = next(document.pages())
    page_img = get_image_from_page(page)
    session = ort.InferenceSession("mnist-8.onnx")
    student_id = extract_student_id(page_img, session)
    if student_id is None:
        print("Could not detect file student id!")
        print("Leaving file ")
        return
    student_id = fix_detected_student_ids(args.known_student_ids_fname, [student_id])[0]
    orig_file = Path(args.file)
    new_file = orig_file.parent / f"{student_id}.pdf"
    print("Writing to", new_file)
    if args.move:
        shutil.move(orig_file, new_file)
    else:
        shutil.copy(orig_file, new_file)


def add_common_options(*parsers):
    for p in parsers:
        p.add_argument(
            "--known-student-ids",
            dest="known_student_ids_fname",
            default=None,
            help="Known student IDs to correct the detected ones.",
        )
        p.add_argument(
            "--first-as-answer-key",
            help="Treat the first attempt as the answer key",
            dest="first_as_key",
            action="store_true",
            default=False,
        )


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser(
        "list", help="Write the detected student ids as a text file"
    )
    list_parser.add_argument("file")
    list_parser.add_argument(
        "-n",
        "--pages-per-attempt",
        dest="pages_per_attempt",
        help="Number of pages per attempt",
        type=int,
        required=True,
    )
    list_parser.add_argument(
        "-o",
        "--output",
        help="The name of the output file",
        dest="out_file",
        default=None,
    )

    split_parser = subparsers.add_parser(
        "split",
        help="Split the file with all attempts into individual attempts with "
        "the corresponding detected student ids as file names.",
    )
    split_parser.add_argument("file", help="The file to split")
    split_parser.add_argument(
        "-n",
        "--pages-per-attempt",
        dest="pages_per_attempt",
        help="Number of pages per attempt",
        type=int,
        required=True,
    )
    split_parser.add_argument(
        "-o",
        "--outdir",
        help="The name of the output directory",
        dest="out_dir",
        default=None,
    )
    add_common_options(split_parser, list_parser)
    args = parser.parse_args()
    if args.command == "list":
        do_list(args)
    elif args.command == "split":
        do_split(args)
    else:
        print("Not a valid command!")


if __name__ == "__main__":
    main()
