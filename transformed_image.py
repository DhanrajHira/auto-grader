import numpy as np
import cv2 as cv


def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)


class ImgTransformationInfo:
    def __init__(self, matrix=None):
        self.matrix = np.identity(3) if matrix is None else matrix

    def record_crop_left(self, pixels):
        assert pixels >= 0, "Cannot crop a negative amount"
        crop_matrix = np.array([[1, 0, -pixels], [0, 1, 0], [0, 0, 1]])
        new_matrix = np.dot(crop_matrix, self.matrix)
        return ImgTransformationInfo(new_matrix)

    def record_crop_right(self, pixels):
        return ImgTransformationInfo(np.array(self.matrix))

    def record_crop_top(self, pixels):
        assert pixels >= 0, "Cannot crop a negative amount"
        crop_matrix = np.array([[1, 0, 0], [0, 1, -pixels], [0, 0, 1]])
        new_matrix = np.dot(crop_matrix, self.matrix)
        return ImgTransformationInfo(new_matrix)

    def record_crop_bottom(self, pixels):
        return ImgTransformationInfo(np.array(self.matrix))

    def record_rotate(self, radians, center):
        cos_rad = np.cos(radians)
        sin_rad = np.sin(radians)
        center_x, center_y = center

        to_origin = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])
        rotation = np.array([[cos_rad, -sin_rad, 0], [sin_rad, cos_rad, 0], [0, 0, 1]])
        from_origin = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])

        rotation_matrix = from_origin @ rotation @ to_origin

        new_matrix = np.dot(rotation_matrix, self.matrix)
        return ImgTransformationInfo(new_matrix)

    def record_scale(self, width_scale, height_scale):
        scale_matrix = np.array(
            [[1 / width_scale, 0, 0], [0, 1 / height_scale, 0], [0, 0, 1]]
        )
        new_matrix = np.dot(scale_matrix, self.matrix)
        return ImgTransformationInfo(new_matrix)

    def to_original(self, x, y):
        inverse_matrix = np.linalg.inv(self.matrix)
        transformed_point = np.array([x, y, 1])
        original_point = np.dot(inverse_matrix, transformed_point)
        return original_point[0], original_point[1]

    def horizontal_length_to_original(self, length):
        # For a forward matrix, the scaling factor is the inverse of what's stored.
        return length / self.matrix[0, 0]

    def vertical_length_to_original(self, length):
        # For a forward matrix, the scaling factor is the inverse of what's stored.
        return length / self.matrix[1, 1]

    def copy(self):
        return ImgTransformationInfo(np.array(self.matrix))


class TransformedImage:
    def __init__(self, orig_img, transform_img=None, transform_info=None):
        self.orig_img = orig_img
        self.transformed_img = (
            transform_img if transform_img is not None else orig_img.copy()
        )
        self.img_transform_info = transform_info or ImgTransformationInfo()

    @property
    def img(self):
        return self.transformed_img

    @property
    def width(self):
        return self.transformed_img.shape[1]

    @property
    def height(self):
        return self.transformed_img.shape[0]

    @property
    def shape(self):
        return self.transformed_img.shape

    def subimage(self, left=None, right=None, top=None, bottom=None):
        left = left or 0
        right = right or self.width
        top = top or 0
        bottom = bottom or self.height
        transform = (
            self.img_transform_info.record_crop_right(right)
            .record_crop_left(left)
            .record_crop_bottom(bottom)
            .record_crop_top(top)
        )
        transformed_img = self.transformed_img[top:bottom, left:right].copy()
        return TransformedImage(self.orig_img, transformed_img, transform)

    def crop_right(self, pixels):
        transform = self.img_transform_info.record_crop_right(pixels)
        transformed_img = self.transformed_img[:, :pixels].copy()
        return TransformedImage(self.orig_img, transformed_img, transform)

    def crop_left(self, pixels):
        transform = self.img_transform_info.record_crop_left(pixels)
        transformed_img = self.transformed_img[:, pixels:].copy()
        return TransformedImage(self.orig_img, transformed_img, transform)

    def crop_top(self, pixels):
        transform = self.img_transform_info.record_crop_top(pixels)
        transformed_img = self.transformed_img[pixels:, :]
        return TransformedImage(self.orig_img, transformed_img, transform)

    def crop_bottom(self, pixels):
        transform = self.img_transform_info.record_crop_bottom(pixels)
        transformed_img = self.transformed_img[:pixels, :]
        return TransformedImage(self.orig_img, transformed_img, transform)

    def scale_to(self, target_width, target_height):
        height, width = self.transformed_img.shape[0], self.transformed_img.shape[1]
        transform = self.img_transform_info.record_scale(
            width / target_width, height / target_height
        )
        transformed_img = cv.resize(
            self.transformed_img,
            (target_width, target_height),
            interpolation=cv.INTER_AREA,
        )
        return TransformedImage(self.orig_img, transformed_img, transform)

    def rotate(self, radians):
        height, width = self.transformed_img.shape[0], self.transformed_img.shape[1]
        center = (width // 2, height // 2)
        transform = self.img_transform_info.record_rotate(radians, center)
        rotation_matrix = cv.getRotationMatrix2D(center, np.degrees(radians), 1.0)
        transformed_img = cv.warpAffine(
            self.transformed_img,
            rotation_matrix,
            (width, height),
            flags=cv.INTER_CUBIC,
            borderMode=cv.BORDER_REPLICATE,
        )
        return TransformedImage(self.orig_img, transformed_img, transform)

    def threshold(self, threshold, threshold_kind=cv.THRESH_BINARY_INV):
        _, transformed_img = cv.threshold(
            self.transformed_img, threshold, 255, threshold_kind
        )
        return TransformedImage(
            self.orig_img, transformed_img, self.img_transform_info.copy()
        )

    def otsu_threshold(self, threshold_kind=cv.THRESH_BINARY_INV):
        _, transformed_img = cv.threshold(
            self.transformed_img, 0, 255, threshold_kind + cv.THRESH_OTSU
        )
        return TransformedImage(
            self.orig_img, transformed_img, self.img_transform_info.copy()
        )

    def gaussian_blur(self, blur_mask):
        transformed_img = cv.GaussianBlur(
            self.transformed_img, (blur_mask, blur_mask), 0
        )
        return TransformedImage(
            self.orig_img, transformed_img, self.img_transform_info.copy()
        )

    def erode(self, kernel_size, iterations=1):
        erosion_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        transformed_img = cv.erode(
            self.transformed_img, erosion_kernel, iterations=iterations
        )
        return TransformedImage(
            self.orig_img, transformed_img, self.img_transform_info.copy()
        )

    def sharpen(self, kernel):
        transformed_img = cv.filter2D(self.transformed_img, -1, kernel)
        return TransformedImage(
            self.orig_img, transformed_img, self.img_transform_info.copy()
        )

    def grayscale(self):
        transformed_img = cv.cvtColor(self.transformed_img, cv.COLOR_BGR2GRAY)
        return TransformedImage(
            self.orig_img, transformed_img, self.img_transform_info.copy()
        )

    def show(self):
        show_image(self.transformed_img)

    def show_original(self):
        show_image(self.orig_img)

    def contours(self, mode, method=cv.CHAIN_APPROX_SIMPLE):
        return cv.findContours(self.img, mode, method)

    @classmethod
    def blank(cls, width, height):
        return TransformedImage(np.zeros((height, width), dtype=np.uint8))
