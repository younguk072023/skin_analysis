from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2lab
from scipy.stats import wasserstein_distance

ROOT = Path(r"C:\Users\park_younguk\Desktop\skin")
CLASSES = ["mild", "moderate", "severe", "very severe"]

LESION_IS_BLACK = True  #마스크의 검은색 병변인 경우

def read_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    return mask

def compute_lab_ita(rgb):
    rgb_norm = rgb.astype(np.float32) / 255.0   #이미지 픽셀 정규화
    lab = rgb2lab(rgb_norm)

    L = lab[:, :, 0]    #픽셀의 밝기
    a = lab[:, :, 1]    #픽셀의 빨강 -> 초록 정보 값
    b = lab[:, :, 2]    #픽셀의 노랑 -> 파랑 정보 값

    ita = np.degrees(np.arctan2((L - 50), b + 1e-8))    #각 픽셀의 ITA 계산
    return L, a, b, ita

def summarize_region(values):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    return np.mean(values), np.std(values), np.median(values)   #평균, 표준편차, 중앙값

def analyze_one_image(img_path, mask_path, severity):   #원본 이미지, 라벨 이미지 경로 , 중증도 라벨
    rgb = read_rgb(img_path)
    mask = read_mask(mask_path)

    h, w = rgb.shape[:2]    #원본이미지의 높이와 너비

    if LESION_IS_BLACK:
        lesion_mask = mask < 128
    else:
        lesion_mask = mask > 128

    background_mask = ~lesion_mask

    lesion_area = int(np.sum(lesion_mask)) 
    background_area = int(np.sum(background_mask))  
    total_area = h * w

    if lesion_area == 0 or background_area == 0:
        return None

    L, a, b, ita = compute_lab_ita(rgb)

    features = {
        "image_id": img_path.stem,
        "severity": severity,
        "lesion_area_px": lesion_area,
        "background_area_px": background_area,
        "lesion_area_ratio": lesion_area / total_area,  #전체 대비 병변 비율
    }

    for name, arr in {
        "L": L,
        "a": a,
        "b": b,
        "ITA": ita,
    }.items():
        lesion_values = arr[lesion_mask]    #병변에 해당하는 픽셀값들만 1차원 리스트로 추출
        bg_values = arr[background_mask]    #배경에 해당하는 픽셀값들만 1차원 리스트로 추출

        lesion_mean, lesion_std, lesion_median = summarize_region(lesion_values)
        bg_mean, bg_std, bg_median = summarize_region(bg_values)

        #ITA 병변 순수 값
        features[f"lesion_{name}_mean"] = lesion_mean
        features[f"lesion_{name}_std"] = lesion_std
        features[f"lesion_{name}_median"] = lesion_median

        #ITA 배경 순수 값
        features[f"bg_{name}_mean"] = bg_mean
        features[f"bg_{name}_std"] = bg_std
        features[f"bg_{name}_median"] = bg_median   

        #단순 차이
        features[f"delta_{name}_mean"] = lesion_mean - bg_mean
        features[f"abs_delta_{name}_mean"] = abs(lesion_mean - bg_mean)

        if len(lesion_values) > 0 and len(bg_values) > 0:
            features[f"wasserstein_{name}"] = wasserstein_distance(
                lesion_values[np.isfinite(lesion_values)],
                bg_values[np.isfinite(bg_values)]
            )
        else:
            features[f"wasserstein_{name}"] = np.nan

    return features

def main():
    rows = []
    missing_labels = []

    for severity in CLASSES:
        data_dir = ROOT / severity / "data"
        label_dir = ROOT / severity / "label"

        image_paths = sorted([
            p for p in data_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        ])

        for img_path in image_paths:
            mask_candidates = list(label_dir.glob(img_path.stem + ".*"))

            if len(mask_candidates) == 0:
                missing_labels.append(str(img_path))
                continue

            mask_path = mask_candidates[0]

            try:
                result = analyze_one_image(img_path, mask_path, severity)
                if result is not None:
                    rows.append(result)
            except Exception as e:
                print(f"[ERROR] {img_path.name}: {e}")

    df = pd.DataFrame(rows)
    out_path = ROOT / "lesion_background_ita_lab_features.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("Saved:", out_path)
    print("Total analyzed images:", len(df))
    print("Missing labels:", len(missing_labels))

    if missing_labels:
        missing_path = ROOT / "missing_labels.txt"
        missing_path.write_text("\n".join(missing_labels), encoding="utf-8")
        print("Missing label list saved:", missing_path)

if __name__ == "__main__":
    main()