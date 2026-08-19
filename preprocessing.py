from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scipy.ndimage import binary_erosion
from scipy.stats import wasserstein_distance

from skimage.color import rgb2lab
from skimage.feature import graycoprops, local_binary_pattern


# ============================================================
# 설정
# ============================================================

ROOT = Path(r"E:\gan_analysis_001612")

CLASSES = [
    "mild",
    "moderate",
    "severe",
    "very_severe",
]

# True  : mask의 검은색 영역 = lesion
# False : mask의 흰색 영역 = lesion
LESION_IS_BLACK = True


# ============================================================
# GLCM 설정
# ============================================================

# Grayscale 0~255를 32단계로 양자화
GLCM_LEVELS = 32

# Pixel distance = 1
# 0°, 45°, 90°, 135°
GLCM_OFFSETS = [
    (0, 1),      # 0°
    (-1, 1),     # 45°
    (-1, 0),     # 90°
    (-1, -1),    # 135°
]


# ============================================================
# LBP 설정
# ============================================================

LBP_POINTS = 8
LBP_RADIUS = 1


TEXTURE_FEATURES = [
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "glcm_entropy",
    "lbp_entropy",
]


# ============================================================
# 이미지 / 마스크 불러오기
# ============================================================

def read_rgb(path):
    """이미지를 RGB 형식으로 불러온다."""

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


def read_mask(path):
    """Segmentation mask를 grayscale로 불러온다."""

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")

    return mask


# ============================================================
# Color: RGB → CIELAB + ITA
# ============================================================

def compute_lab_ita(rgb):
    """
    RGB 이미지를 CIELAB으로 변환하고
    픽셀 단위 ITA를 계산한다.

    L*  : 밝기
    a*  : Green ↔ Red
    b*  : Blue ↔ Yellow
    ITA : Individual Typology Angle
    """

    rgb_float = (
        rgb.astype(np.float32) / 255.0
    )

    lab = rgb2lab(rgb_float)

    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    ita = np.degrees(
        np.arctan2(
            L - 50.0,
            b + 1e-8
        )
    )

    return {
        "L": L,
        "a": a,
        "b": b,
        "ITA": ita,
    }


# ============================================================
# Color Feature
# ============================================================

def extract_color_features(
    color_maps,
    lesion_mask,
    background_mask
):
    """
    L*, a*, b*, ITA 각각에 대해:

    1. lesion mean
    2. background mean
    3. lesion - background
    4. |lesion - background|
    5. Wasserstein distance

    를 계산한다.

    Background가 존재하지 않는 경우:
        lesion mean은 계산
        background / delta / abs_delta / Wasserstein은 NaN
    """

    features = {}

    for name, channel in color_maps.items():

        # ----------------------------------------------------
        # Lesion / Background pixel 추출
        # ----------------------------------------------------

        lesion_values = channel[
            lesion_mask
        ]

        bg_values = channel[
            background_mask
        ]


        # NaN / Inf 제거
        lesion_values = lesion_values[
            np.isfinite(lesion_values)
        ]

        bg_values = bg_values[
            np.isfinite(bg_values)
        ]


        # ====================================================
        # Lesion
        # ====================================================

        if lesion_values.size > 0:

            lesion_mean = float(
                np.mean(lesion_values)
            )

        else:

            lesion_mean = np.nan


        features[
            f"lesion_{name}_mean"
        ] = lesion_mean


        # ====================================================
        # Background가 없는 경우
        # ====================================================

        if bg_values.size == 0:

            features[
                f"bg_{name}_mean"
            ] = np.nan

            features[
                f"delta_{name}_mean"
            ] = np.nan

            features[
                f"abs_delta_{name}_mean"
            ] = np.nan

            features[
                f"wasserstein_{name}"
            ] = np.nan

            continue


        # ====================================================
        # Background mean
        # ====================================================

        bg_mean = float(
            np.mean(bg_values)
        )

        features[
            f"bg_{name}_mean"
        ] = bg_mean


        # ====================================================
        # Lesion - Background
        # ====================================================

        if np.isfinite(lesion_mean):

            delta = (
                lesion_mean
                - bg_mean
            )

            features[
                f"delta_{name}_mean"
            ] = delta

            features[
                f"abs_delta_{name}_mean"
            ] = abs(delta)


            # ------------------------------------------------
            # Wasserstein distance
            # ------------------------------------------------

            features[
                f"wasserstein_{name}"
            ] = float(
                wasserstein_distance(
                    lesion_values,
                    bg_values
                )
            )

        else:

            features[
                f"delta_{name}_mean"
            ] = np.nan

            features[
                f"abs_delta_{name}_mean"
            ] = np.nan

            features[
                f"wasserstein_{name}"
            ] = np.nan


    return features


# ============================================================
# Texture: RGB → Grayscale
# ============================================================

def rgb_to_gray(rgb):
    """
    RGB 원본 이미지를 Grayscale로 변환한다.

    Texture는 CIELAB L*가 아니라
    이 Grayscale 영상에서 계산한다.
    """

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )


# ============================================================
# Grayscale 양자화
# ============================================================

def quantize_gray(
    gray,
    levels=GLCM_LEVELS
):
    """
    Grayscale 0~255를
    0~31 범위로 양자화한다.

    GLCM_LEVELS = 32
    """

    gray = gray.astype(
        np.float32
    )

    quantized = np.floor(
        gray / 256.0 * levels
    )

    quantized = np.clip(
        quantized,
        0,
        levels - 1
    )

    return quantized.astype(
        np.uint8
    )


# ============================================================
# GLCM용 Masked Pixel Pair
# ============================================================

def get_valid_pixel_pairs(
    image,
    region_mask,
    dy,
    dx
):
    """
    두 픽셀이 모두 같은 ROI 내부에 존재할 때만
    GLCM 계산에 사용한다.

    lesion → lesion      : 사용
    lesion → background  : 제외
    """

    h, w = image.shape


    # --------------------------------------------------------
    # Y 방향
    # --------------------------------------------------------

    if dy >= 0:

        y1 = slice(
            0,
            h - dy
        )

        y2 = slice(
            dy,
            h
        )

    else:

        y1 = slice(
            -dy,
            h
        )

        y2 = slice(
            0,
            h + dy
        )


    # --------------------------------------------------------
    # X 방향
    # --------------------------------------------------------

    if dx >= 0:

        x1 = slice(
            0,
            w - dx
        )

        x2 = slice(
            dx,
            w
        )

    else:

        x1 = slice(
            -dx,
            w
        )

        x2 = slice(
            0,
            w + dx
        )


    # --------------------------------------------------------
    # Pixel pair
    # --------------------------------------------------------

    source = image[
        y1,
        x1
    ]

    target = image[
        y2,
        x2
    ]


    source_mask = region_mask[
        y1,
        x1
    ]

    target_mask = region_mask[
        y2,
        x2
    ]


    # 두 픽셀이 모두 ROI 내부인 경우만 사용
    valid = (
        source_mask
        & target_mask
    )


    return (
        source[valid],
        target[valid]
    )


# ============================================================
# GLCM Feature
# ============================================================

def compute_glcm_features(
    gray,
    region_mask
):
    """
    Grayscale 영상에서 masked GLCM을 계산한다.

    방향:
        0°
        45°
        90°
        135°

    각 방향에서 계산한 후
    네 방향 평균을 최종 feature로 사용한다.
    """

    image = quantize_gray(
        gray
    )


    properties = {
        "contrast": [],
        "homogeneity": [],
        "energy": [],
        "correlation": [],
        "entropy": [],
    }


    # ========================================================
    # 방향별 GLCM 계산
    # ========================================================

    for dy, dx in GLCM_OFFSETS:

        source, target = (
            get_valid_pixel_pairs(
                image,
                region_mask,
                dy,
                dx
            )
        )


        if source.size == 0:
            continue


        # ----------------------------------------------------
        # 32 × 32 GLCM 생성
        # ----------------------------------------------------

        glcm = np.zeros(
            (
                GLCM_LEVELS,
                GLCM_LEVELS
            ),
            dtype=np.float64
        )


        # Source → Target
        np.add.at(
            glcm,
            (source, target),
            1
        )


        # Target → Source
        # symmetric GLCM
        np.add.at(
            glcm,
            (target, source),
            1
        )


        if glcm.sum() == 0:
            continue


        # ----------------------------------------------------
        # graycoprops 입력 형식:
        #
        # levels × levels × distance × angle
        # ----------------------------------------------------

        glcm_4d = glcm[
            :,
            :,
            None,
            None
        ]


        # ----------------------------------------------------
        # Contrast
        # ----------------------------------------------------

        contrast = graycoprops(
            glcm_4d,
            "contrast"
        )[0, 0]

        properties[
            "contrast"
        ].append(
            float(contrast)
        )


        # ----------------------------------------------------
        # Homogeneity
        # ----------------------------------------------------

        homogeneity = graycoprops(
            glcm_4d,
            "homogeneity"
        )[0, 0]

        properties[
            "homogeneity"
        ].append(
            float(homogeneity)
        )


        # ----------------------------------------------------
        # Energy
        # ----------------------------------------------------

        energy = graycoprops(
            glcm_4d,
            "energy"
        )[0, 0]

        properties[
            "energy"
        ].append(
            float(energy)
        )


        # ----------------------------------------------------
        # Correlation
        # ----------------------------------------------------

        correlation = graycoprops(
            glcm_4d,
            "correlation"
        )[0, 0]

        properties[
            "correlation"
        ].append(
            float(correlation)
        )


        # ----------------------------------------------------
        # Entropy
        # ----------------------------------------------------

        probability = (
            glcm / glcm.sum()
        )

        probability = probability[
            probability > 0
        ]


        entropy = -np.sum(
            probability
            * np.log2(probability)
        )


        properties[
            "entropy"
        ].append(
            float(entropy)
        )


    # ========================================================
    # 4개 방향 평균
    # ========================================================

    output = {}


    for name, values in properties.items():

        if len(values) > 0:

            output[
                f"glcm_{name}"
            ] = float(
                np.mean(values)
            )

        else:

            output[
                f"glcm_{name}"
            ] = np.nan


    return output


# ============================================================
# LBP Feature
# ============================================================

def compute_lbp_features(
    gray,
    region_mask
):
    """
    Grayscale 영상에서 uniform LBP를 계산한다.

    최종 feature:
        LBP histogram entropy
    """

    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform"
    )


    # --------------------------------------------------------
    # ROI 경계 제거
    #
    # 경계 근처에서 ROI 바깥 픽셀이
    # LBP 계산에 포함되는 것을 줄인다.
    # --------------------------------------------------------

    interior_mask = binary_erosion(
        region_mask,
        structure=np.ones(
            (3, 3),
            dtype=bool
        ),
        iterations=LBP_RADIUS
    )


    lbp_values = lbp[
        interior_mask
    ]


    if lbp_values.size == 0:

        return {
            "lbp_entropy": np.nan
        }


    # --------------------------------------------------------
    # Uniform LBP histogram
    #
    # P = 8
    # → P + 2 = 10개 category
    # --------------------------------------------------------

    histogram, _ = np.histogram(
        lbp_values,
        bins=np.arange(
            LBP_POINTS + 3
        )
    )


    probability = histogram.astype(
        np.float64
    )


    if probability.sum() == 0:

        return {
            "lbp_entropy": np.nan
        }


    probability /= (
        probability.sum()
    )


    probability = probability[
        probability > 0
    ]


    entropy = -np.sum(
        probability
        * np.log2(probability)
    )


    return {
        "lbp_entropy": float(entropy)
    }


# ============================================================
# GLCM + LBP 통합
# ============================================================

def compute_texture_features(
    gray,
    region_mask
):
    """
    하나의 ROI에서
    GLCM + LBP texture feature를 계산한다.
    """

    features = {}


    features.update(
        compute_glcm_features(
            gray,
            region_mask
        )
    )


    features.update(
        compute_lbp_features(
            gray,
            region_mask
        )
    )


    return features


# ============================================================
# Lesion / Background Texture
# ============================================================

def extract_texture_features(
    gray,
    lesion_mask,
    background_mask
):
    """
    병변과 배경의 texture를 각각 계산한 뒤

    1. lesion
    2. background
    3. lesion - background

    를 저장한다.

    Texture에서는 abs_delta는 사용하지 않는다.
    """

    lesion_texture = (
        compute_texture_features(
            gray,
            lesion_mask
        )
    )


    bg_texture = (
        compute_texture_features(
            gray,
            background_mask
        )
    )


    features = {}


    for name in TEXTURE_FEATURES:

        lesion_value = (
            lesion_texture[name]
        )

        bg_value = (
            bg_texture[name]
        )


        # ----------------------------------------------------
        # Lesion
        # ----------------------------------------------------

        features[
            f"lesion_{name}"
        ] = lesion_value


        # ----------------------------------------------------
        # Background
        # ----------------------------------------------------

        features[
            f"bg_{name}"
        ] = bg_value


        # ----------------------------------------------------
        # Lesion - Background
        # ----------------------------------------------------

        if (
            np.isfinite(lesion_value)
            and np.isfinite(bg_value)
        ):

            features[
                f"delta_{name}"
            ] = (
                lesion_value
                - bg_value
            )

        else:

            features[
                f"delta_{name}"
            ] = np.nan


    return features


# ============================================================
# 이미지 한 장 Feature Extraction
# ============================================================

def analyze_one_image(
    img_path,
    mask_path,
    severity
):
    """
    한 RGB 이미지에서

    Color feature
    +
    Texture feature

    를 추출한다.
    """

    # --------------------------------------------------------
    # 이미지 / Mask 불러오기
    # --------------------------------------------------------

    rgb = read_rgb(
        img_path
    )

    mask = read_mask(
        mask_path
    )


    height, width = (
        rgb.shape[:2]
    )


    # --------------------------------------------------------
    # 이미지 / Mask 크기 확인
    # --------------------------------------------------------

    if mask.shape != (
        height,
        width
    ):

        raise ValueError(
            f"Size mismatch: "
            f"{img_path.name}, "
            f"image={rgb.shape[:2]}, "
            f"mask={mask.shape}"
        )


    # ========================================================
    # Lesion / Background Mask
    # ========================================================

    if LESION_IS_BLACK:

        lesion_mask = (
            mask < 128
        )

    else:

        lesion_mask = (
            mask > 128
        )


    background_mask = (
        ~lesion_mask
    )


    # --------------------------------------------------------
    # QC용 영역 확인
    #
    # 면적 자체는 feature로 저장하지 않는다.
    # --------------------------------------------------------

    lesion_pixels = int(
        lesion_mask.sum()
    )

    background_pixels = int(
        background_mask.sum()
    )


    if (
        lesion_pixels == 0
    ):

        raise ValueError(
            "Lesion region is empty."
        )


    # ========================================================
    # 기본 정보
    # ========================================================

    features = {
        "image_id":
            img_path.stem,

        "severity":
            severity,
    }


    # ========================================================
    # COLOR
    #
    # RGB
    # ↓
    # CIELAB
    # ↓
    # L*, a*, b*, ITA
    # ========================================================

    color_maps = compute_lab_ita(
        rgb
    )


    color_features = (
        extract_color_features(
            color_maps,
            lesion_mask,
            background_mask
        )
    )


    features.update(
        color_features
    )


    # ========================================================
    # TEXTURE
    #
    # RGB
    # ↓
    # Grayscale
    # ↓
    # GLCM + LBP
    # ========================================================

    gray = rgb_to_gray(
        rgb
    )


    texture_features = (
        extract_texture_features(
            gray,
            lesion_mask,
            background_mask
        )
    )


    features.update(
        texture_features
    )


    return features


# ============================================================
# Main
# ============================================================

def main():

    rows = []

    missing_labels = []
    errors = []


    # ========================================================
    # Severity 반복
    # ========================================================

    for severity in CLASSES:

        data_dir = (
            ROOT
            / severity
            / "data"
        )

        label_dir = (
            ROOT
            / severity
            / "label"
        )


        # ----------------------------------------------------
        # 폴더 확인
        # ----------------------------------------------------

        if not data_dir.exists():

            print(
                f"[WARNING] "
                f"Missing data directory: "
                f"{data_dir}"
            )

            continue


        if not label_dir.exists():

            print(
                f"[WARNING] "
                f"Missing label directory: "
                f"{label_dir}"
            )

            continue


        # ----------------------------------------------------
        # 이미지 목록
        # ----------------------------------------------------

        image_paths = sorted([
            path
            for path in data_dir.iterdir()
            if path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
            }
        ])


        print(
            f"[{severity}] "
            f"{len(image_paths)} images"
        )


        # ====================================================
        # 이미지 반복
        # ====================================================

        for img_path in image_paths:

            # ------------------------------------------------
            # 같은 이름의 Mask 검색
            # ------------------------------------------------

            mask_candidates = list(
                label_dir.glob(
                    img_path.stem + ".*"
                )
            )


            if not mask_candidates:

                missing_labels.append(
                    str(img_path)
                )

                continue


            mask_path = (
                mask_candidates[0]
            )


            # ------------------------------------------------
            # Feature Extraction
            # ------------------------------------------------

            try:

                result = analyze_one_image(
                    img_path,
                    mask_path,
                    severity
                )


                rows.append(
                    result
                )


            except Exception as e:

                print(
                    f"[ERROR] "
                    f"{img_path.name}: "
                    f"{e}"
                )


                errors.append(
                    f"{img_path}\t{e}"
                )


    # ========================================================
    # DataFrame 생성
    # ========================================================

    df = pd.DataFrame(
        rows
    )


    # ========================================================
    # CSV 저장
    # ========================================================

    output_path = (
        ROOT
        / "gan_lesion_background_color_texture_features.csv"
    )


    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # 결과 출력
    # ========================================================

    print()
    print("=" * 60)
    print("Feature extraction completed")
    print("=" * 60)

    print(
        f"Analyzed images : "
        f"{len(df)}"
    )

    print(
        f"Missing labels  : "
        f"{len(missing_labels)}"
    )

    print(
        f"Errors          : "
        f"{len(errors)}"
    )

    print(
        f"Total columns   : "
        f"{len(df.columns)}"
    )

    print(
        f"CSV             : "
        f"{output_path}"
    )


    # --------------------------------------------------------
    # 기대 column 수 확인
    #
    # image_id + severity
    # + Color 20
    # + Texture 18
    # = 40
    # --------------------------------------------------------

    if not df.empty:

        expected_columns = 40

        if len(df.columns) == expected_columns:

            print(
                "Column check    : OK "
                "(40 columns)"
            )

        else:

            print(
                f"Column check    : "
                f"Expected 40, "
                f"found {len(df.columns)}"
            )


    # ========================================================
    # Missing Label Log
    # ========================================================

    if missing_labels:

        missing_path = (
            ROOT
            / "missing_labels.txt"
        )

        missing_path.write_text(
            "\n".join(
                missing_labels
            ),
            encoding="utf-8"
        )


    # ========================================================
    # Error Log
    # ========================================================

    if errors:

        error_path = (
            ROOT
            / "feature_extraction_errors.txt"
        )

        error_path.write_text(
            "\n".join(
                errors
            ),
            encoding="utf-8"
        )


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()