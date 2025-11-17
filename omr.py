import cv2 as cv
import pymupdf
import numpy as np

from transformed_image import ImgTransformationInfo, TransformedImage

from dataclasses import dataclass
import logging
from itertools import chain

BGR_BLUE = (255, 0, 0)
BGR_GREEN = (0, 255, 0)

PDF_RED = (1.0, 0.0, 0.0)
PDF_GREEN = (0.0, 1.0, 0.0)
PDF_BLUE = (0.0, 0.0, 1.0)

DPI = 200
TRIANGLE_MIN_AREA = DPI
TARGET_IMG_HEIGHT = 2200
TARGET_IMG_WIDTH = 1700
TARGET_ASPECT_RATIO = TARGET_IMG_WIDTH / TARGET_IMG_HEIGHT

logger = logging.getLogger("OMR")


@dataclass
class Bubble:
    x: int
    y: int
    radius: int

    @property
    def cords(self):
        return (self.x, self.y)

    def draw(self, img):
        cv.circle(img, self.cords, self.radius, BGR_GREEN)

    def to_pdf_cords(self, transform: ImgTransformationInfo):
        assert transform is not None
        x_orig, y_orig = transform.to_original(self.x, self.y)

        # 3. Scale to PDF coordinates
        pdf_x = x_orig * 72 / DPI
        pdf_y = y_orig * 72 / DPI
        pdf_radius = max(
            transform.horizontal_length_to_original(self.radius),
            transform.vertical_length_to_original(self.radius),
        )
        pdf_radius = int(pdf_radius * 72 / DPI)

        return Bubble(x=pdf_x, y=pdf_y, radius=pdf_radius)


@dataclass
class OrientationLine:
    start_x: int
    start_y: int
    end_x: int
    end_y: int

    @property
    def start_cords(self):
        return self.start_x, self.start_y

    @property
    def end_cords(self):
        return self.end_x, self.end_y

    @property
    def angle(self):
        return np.arctan2(self.end_y - self.start_y, self.end_x - self.start_x)

    def draw(self, img: TransformedImage):
        cv.line(img.img, self.start_cords, self.end_cords, BGR_BLUE, 2)


@dataclass
class GuideMark:
    x: float
    y: float
    width: float
    height: float

    @property
    def top_left_cords(self):
        return (self.x, self.y)

    @property
    def bottom_right_cords(self):
        return (self.x + self.width, self.y + self.height)

    @property
    def center_cords(self):
        return (self.center_x, self.center_y)

    @property
    def center_x(self):
        return self.x + self.width / 2

    @property
    def center_y(self):
        return self.y + self.height / 2

    def draw(self, img):
        cv.rectangle(
            img,
            (int(self.x), int(self.y)),
            (int(self.x + self.width), int(self.y + self.height)),
            BGR_BLUE,
        )

    def to_pdf_cords(self, transform: ImgTransformationInfo):
        assert transform is not None
        x_orig, y_orig = transform.to_original(self.x, self.y)

        # 3. Scale to PDF coordinates
        pdf_x = x_orig * 72 / DPI
        pdf_y = y_orig * 72 / DPI

        pdf_w = transform.horizontal_length_to_original(self.width)
        pdf_h = transform.vertical_length_to_original(self.height)
        pdf_w = pdf_w * 72 / DPI
        pdf_h = pdf_h * 72 / DPI
        return GuideMark(x=pdf_x, y=pdf_y, width=pdf_w, height=pdf_h)

    def shifted_by(self, x_offset, y_offset):
        return GuideMark(
            x=self.x + x_offset,
            y=self.y + y_offset,
            width=self.width,
            height=self.height,
        )


def detect_horizontal_and_vertical_guides(all_guides, tolerance=15):
    # Since there are always going to be more questions than options
    # first we figure out the x coordinate of the horizontal guides
    assert len(all_guides) > 2
    horizontal_guide_x = sorted([guide.x for guide in all_guides])[len(all_guides) // 2]
    horizontal_guides = []
    vertical_guides = []
    for guide in all_guides:
        if abs(guide.x - horizontal_guide_x) < tolerance:
            horizontal_guides.append(guide)
        else:
            vertical_guides.append(guide)
    return vertical_guides, horizontal_guides


class GuideMatrix:
    def __init__(self, guide_points: list):
        v_guides, h_guides = detect_horizontal_and_vertical_guides(guide_points)
        logger.debug(f"Detected a grid {len(v_guides)}x{len(h_guides)} grid")
        self.vertical_guides = v_guides
        self.vertical_guides.sort(key=lambda g: g.x)
        self.horizontal_guides = h_guides
        self.horizontal_guides.sort(key=lambda g: g.y)

    def cells_centers(self):
        for vert_guide in self.vertical_guides:
            for horizontal_guide in self.horizontal_guides:
                yield (vert_guide.center_x, horizontal_guide.center_y)

    @property
    def num_rows(self):
        return len(self.horizontal_guides)

    @property
    def num_cols(self):
        return len(self.vertical_guides)

    def cell_center_at(self, row, col):
        vert_guide = self.vertical_guides[col]
        hor_guide = self.horizontal_guides[row]
        return (vert_guide.center_x, hor_guide.center_y)

    def to_pdf_cords(self, transform: ImgTransformationInfo):
        pdf_guides = [g.to_pdf_cords(transform) for g in self.vertical_guides] + [
            g.to_pdf_cords(transform) for g in self.horizontal_guides
        ]
        return GuideMatrix(pdf_guides)

    def horizontal_guide_for_row(self, row):
        return self.horizontal_guides[row]

    def __repr__(self):
        return f"GuideMatrix<{self.num_rows}x{self.num_cols}>(<{self.horizontal_guides}>x<{self.vertical_guides}>)"

    def is_point_in_a_cell(self, x, y, tolerance=30):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_x, cell_y = self.cell_center_at(row, col)
                x_diff = abs(cell_x - x)
                y_diff = abs(cell_y - y)
                if x_diff < tolerance and y_diff < tolerance:
                    return True
        return False

    def draw(self, img):
        for guide in self.horizontal_guides:
            guide.draw(img)

        for guide in self.vertical_guides:
            guide.draw(img)


def get_line_angle(line):
    x1, y1, x2, y2 = line[0]
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def fix_page_orientation(page_img: TransformedImage):
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = page_img.grayscale().threshold(100).erode(5).sharpen(sharpening_kernel)

    contours, _ = img.contours(cv.RETR_EXTERNAL)

    h, w = img.shape[:2]
    min_line_length = int(w * 0.80)

    lines = []
    for contour in contours:
        _, _, c_w, c_h = cv.boundingRect(contour)
        if c_w > min_line_length and c_h < 50:  # Filter for long, horizontal contours
            # Fit a line to the contour points
            [vx, vy, x, y] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
            # Extrapolate the line to the image boundaries
            lefty = int((-x * vy / vx) + y)
            righty = int(((w - x) * vy / vx) + y)
            lines.append(
                OrientationLine(start_x=0, start_y=lefty, end_x=w - 1, end_y=righty)
            )

    if not lines:
        logger.warning(
            "No dominant long lines were detected. Returning original image."
        )
        return page_img

    for line in lines:
        line.draw(page_img)

    angles = [line.angle for line in lines]
    median_angle = float(np.median(angles))
    logger.debug(f"Rotating image by {-median_angle} degrees")

    top_line = min(lines, key=lambda line: line.start_y)
    top = max(top_line.start_y, top_line.end_y)
    bottom_line = max(lines, key=lambda line: line.start_y)
    bottom = min(bottom_line.start_y, bottom_line.end_y)
    logger.debug(f"Cropping image to height: {top}, {bottom}")
    page_img = page_img.rotate(median_angle).crop_top(top).crop_bottom(bottom)
    # page_img.show()
    # page_img.show_original()
    return page_img


def detect_triangles(img: TransformedImage, min_area=DPI + 50):
    # Need to use a lower threshold here because students sometimes squible light
    # marks around the triangles.
    img = img.grayscale().gaussian_blur(5).threshold(100)
    contours, hierarchy = img.contours(cv.RETR_TREE)
    all_triangles = []

    for i, contour in enumerate(contours):
        # Filter out very small contours that might be noise
        if cv.contourArea(contour) < min_area:
            continue

        # Approximate the contour to a polygon.
        # The epsilon parameter is key; it determines how "closely" the
        # polygon must match the contour. A smaller value means a closer match.
        # We use a percentage of the contour's perimeter.
        epsilon = 0.04 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        # A triangle will have 3 vertices
        if len(approx) != 3:
            continue
        # Check if it's a filled shape by ensuring it has no child contours.
        # hierarchy[0][i][2] is the index of the first child contour.
        # It's -1 if there are no children.
        if hierarchy[0][i][2] != -1:
            continue
        x, y, w, h = cv.boundingRect(contour)
        all_triangles.append(GuideMark(x=x, y=y, width=w, height=h))

    return all_triangles


def detect_bubbles(
    img: TransformedImage,
    min_area=DPI,
    circularity_threshold=0.8,
    solidity_threshold=0.8,
):
    # Need big blur mask and threshold because students sometimes don't pencil
    # in the mark enough or the scanner makes their marks look jagged.
    img = img.grayscale().gaussian_blur(17).threshold(230)
    # img.show()
    contours, hierarchy = img.contours(cv.RETR_CCOMP)
    detected_circles = []
    for i, contour in enumerate(contours):
        # We are looking for contours that are external (no parent) and have no children.
        # This indicates a solid, filled shape.
        # hierarchy[0][i][3] is the parent index. -1 means no parent (i.e., external).
        # hierarchy[0][i][2] is the first child index. -1 means no child.
        child = hierarchy[0][i][2]
        parent = hierarchy[0][i][3]
        is_filled_shape = parent == -1 and child == -1

        if not is_filled_shape:
            logging.info("Skipping because its not a filled image")
            continue

        # This contour is a candidate for a filled shape.
        # Now, apply the geometric filters to see if it's a circle.

        # a) Filter by area to remove small noise
        area = cv.contourArea(contour)
        if area < min_area:
            logging.info("Skipping because area less than min area")
            continue

        # b) Calculate circularity to filter non-circular shapes
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            logging.info("Skipping because parameter is 0")
            continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        if circularity < circularity_threshold:
            logging.info(
                f"Skipping because circularity is less than threshold {circularity}vs{circularity_threshold}"
            )
            continue

        # c) The solidity check is now a secondary check for convexity.
        # Solidity = Contour Area / Convex Hull Area
        # A filled circle will have a solidity close to 1.
        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)
        if hull_area == 0:
            logging.info("Skipping because hull area is zero")
            continue

        solidity = float(area) / hull_area
        if solidity < solidity_threshold:
            logging.info(
                f"Skipping because solidity is less than threshold {solidity}vs{solidity_threshold}"
            )
            continue

        # This contour is a good candidate for a filled circle
        # Get the enclosing circle
        ((x, y), radius) = cv.minEnclosingCircle(contour)
        detected_circles.append(Bubble(x=int(x), y=int(y), radius=int(radius)))
    return detected_circles


def has_a_bubble_at(point, bubbles, tolerance=30):
    for bubble in bubbles:
        if (
            abs(point[0] - bubble.x) < tolerance
            and abs(point[1] - bubble.y) < tolerance
        ):
            return True
    return False


def raw_bubbles_to_attempt_matrix(bubbles, guide_matrix):
    num_rows, num_cols = guide_matrix.num_rows, guide_matrix.num_cols
    attempt_matrix = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

    for row in range(num_rows):
        for col in range(num_cols):
            cell_center = guide_matrix.cell_center_at(row, col)
            if has_a_bubble_at(cell_center, bubbles):
                attempt_matrix[row][col] = 1
    return attempt_matrix


def draw_all_objects_on(to_draw_on, *objs_to_draw):
    for obj in objs_to_draw:
        obj.draw(to_draw_on)


def get_image_from_page(page):
    page_image_bytes = page.get_pixmap(dpi=DPI).pil_tobytes(format="png")
    page_image = cv.imdecode(
        np.frombuffer(page_image_bytes, dtype=np.uint8), cv.IMREAD_UNCHANGED
    )
    assert page_image is not None
    page_image = TransformedImage(page_image)
    logger.debug(f"Extracted image from page with resolution: {page_image.shape}")
    page_image = page_image.scale_to(TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT)
    logger.debug(f"Resized to resolution: {(TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT)}")
    return page_image


def pages_to_columns(pages):
    for page in pages:
        page_image = get_image_from_page(page)
        page_image = fix_page_orientation(page_image)
        center = page_image.width // 2
        yield page_image.crop_right(center)
        yield page_image.crop_left(center)


def get_attempts_on_column(img: TransformedImage):
    guides = detect_triangles(img)
    bubbles = detect_bubbles(img)

    # draw_all_objects_on(processed_image, *guides, *bubbles)
    # show_image(processed_image)
    logger.debug(f"Detected {len(guides)} guides")

    guide_matrix = GuideMatrix(guides)
    attempt_matrix = raw_bubbles_to_attempt_matrix(bubbles, guide_matrix)
    bubbles = filter_out_of_grid_bubbles(guide_matrix, bubbles)
    return attempt_matrix, guides, bubbles, guide_matrix


def get_answers_from_file(document):
    all_answers_columns = []
    for column in pages_to_columns(document.pages()):
        answers, _, _, _ = get_attempts_on_column(column)
        all_answers_columns.append(answers)
    return all_answers_columns


def draw_correct_answers_on_page(guide_matrix, answers, radius, page):
    rot_matrix = page.derotation_matrix
    for row, answer in enumerate(answers):
        for col, option in enumerate(answer):
            if option != 1:
                continue
            center = pymupdf.Point(guide_matrix.cell_center_at(row, col))
            center = center * rot_matrix
            page.draw_circle(center, radius, color=PDF_GREEN)


def draw_question_status_on_page(guide_matrix, results, page):
    rot_matrix = page.derotation_matrix
    for row, result in enumerate(results):
        guide = guide_matrix.horizontal_guide_for_row(row)
        guide = guide.shifted_by(-guide.width - 2, 0)
        rect = pymupdf.Rect(
            guide.x, guide.y, guide.x + guide.width, guide.y + guide.height
        )
        rect = rect * rot_matrix
        rect.normalize()
        color = PDF_GREEN if result else PDF_RED
        page.draw_rect(rect, color=color, fill=color)


def filter_out_of_grid_bubbles(guide, bubbles):
    filtered = filter(
        lambda bubble: guide.is_point_in_a_cell(bubble.x, bubble.y, bubble.radius),
        bubbles,
    )
    return list(filtered)


def draw_detected_objects_on_page(bubbles, guides, page):
    rot_matrix = page.derotation_matrix
    for bubble in bubbles:
        center = pymupdf.Point(bubble.x, bubble.y)
        center = center * rot_matrix
        page.draw_circle(
            center,
            bubble.radius,
            color=PDF_RED,
        )

    for guide in guides:
        rect = pymupdf.Rect(
            guide.x, guide.y, guide.x + guide.width, guide.y + guide.height
        )
        rect = rect * rot_matrix
        rect.normalize()
        page.draw_rect(rect, color=PDF_BLUE)


def correct_attempt_positions(attempts, answers):
    positions = []
    for attempt, answer in zip(attempts, answers):
        # for now, the first question with no correct answers marks, the end
        # of the questions
        if not any(answer):
            break
        positions.append(answer == attempt)
    return positions


def calculate_final_score(attempt_columns, answer_columns):
    positions = correct_attempt_positions(
        chain(*attempt_columns), chain(*answer_columns)
    )
    return sum(positions), len(positions)


def repeat_n_times(iterator, n):
    for i in iterator:
        for _ in range(n):
            yield i


def mark_pages(attempt_pages, answer_columns):
    attempt_pages = list(attempt_pages)
    assert len(attempt_pages) > 0, "Attempt needs to have atleast one page"

    all_attempt_columns = []
    for column, column_answers, page in zip(
        pages_to_columns(attempt_pages),
        answer_columns,
        repeat_n_times(attempt_pages, 2),
    ):
        attempts, guides, bubbles, guide_matrix = get_attempts_on_column(column)
        pdf_bubbles = [b.to_pdf_cords(column.img_transform_info) for b in bubbles]
        pdf_guides = [g.to_pdf_cords(column.img_transform_info) for g in guides]
        pdf_guide_matrix = guide_matrix.to_pdf_cords(column.img_transform_info)
        bubble_radius = (
            pdf_bubbles[0].radius + 3 if pdf_bubbles else pdf_guides[0].width
        )
        # guide_matrix.draw(column.img)
        # column.show()

        draw_detected_objects_on_page(pdf_bubbles, pdf_guides, page)
        draw_correct_answers_on_page(
            pdf_guide_matrix,
            column_answers,
            bubble_radius,
            page,
        )
        correct_positions = correct_attempt_positions(column_answers, attempts)
        draw_question_status_on_page(pdf_guide_matrix, correct_positions, page)
        all_attempt_columns.append(attempts)

    logger.info("Found Answer Matrix as:")
    for i, answer in enumerate(answer_columns, start=1):
        logger.info(f" {i:2}: {answer}")

    logger.info("Attempt:")
    for i, attempted_answer in enumerate(all_attempt_columns, start=1):
        logger.info(f" {i:2}: {attempted_answer}")

    score, total_answers = calculate_final_score(all_attempt_columns, answer_columns)
    score_str = f"Score: {score}/{total_answers}"
    logger.info(f"Score: {score_str}")

    first_page = next(iter(attempt_pages))
    score_loc = pymupdf.Point(20, 20)
    score_loc = score_loc * first_page.derotation_matrix
    first_page.insert_text(
        score_loc, score_str, fontsize=24, rotate=first_page.rotation
    )
    return score, total_answers


def chunked(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


def mark_single_file(attempt_file_bytes, answer_file_bytes):
    attempt_file = pymupdf.Document(stream=attempt_file_bytes)
    answer_file = pymupdf.Document(stream=answer_file_bytes)
    logger.debug("Start processing answer file")
    answers = get_answers_from_file(answer_file)
    attempt_pages = list(attempt_file.pages())
    assert len(attempt_pages) % answer_file.page_count == 0, (
        "Not all attempts seem to have all pages"
    )
    scores = []
    attempts = chunked(attempt_pages, answer_file.page_count)
    for num, attempt in enumerate(attempts, start=1):
        try:
            score, total_answers = mark_pages(attempt, answers)
            scores.append((score, total_answers))
        except Exception as _:
            print(f"Failed to mark attempt {num}. Please check the file!")
            scores.append((0, 0))

    document = attempt_file.tobytes()
    attempt_file.close()
    return scores, document


def mark_file(attempt_file_bytes, answer_file_bytes):
    attempt_file = pymupdf.Document(stream=attempt_file_bytes)
    answer_file = pymupdf.Document(stream=answer_file_bytes)
    logger.debug("Start processing answer file")
    answers = get_answers_from_file(answer_file)
    logger.debug("Start processing attempt file")
    score, total_answers = mark_pages(attempt_file.pages(), answers)
    document = attempt_file.tobytes()
    attempt_file.close()
    return (score, total_answers), document
