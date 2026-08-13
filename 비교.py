from pathlib import Path
from PIL import Image
import pandas as pd


# ============================================================
# 1. 기본 설정
# ============================================================

ROOT = Path(r"C:\Users\park_younguk\Desktop\skin")

TARGET_SIZE = (256, 256)

# 원본 폴더명 -> 새 폴더명
CLASS_FOLDERS = {
    "mild": "mild1",
    "moderate": "moderate1",
    "severe": "severe1",
    "very severe": "very severe1",
}

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# 2. 같은 이름의 label 찾기
#    image가 xxx.jpg이고 label이 xxx.png여도 찾도록 함
# ============================================================

def find_label(label_dir, image_path):

    # 완전히 같은 파일명이 있으면 우선 사용
    exact_path = label_dir / image_path.name

    if exact_path.exists():
        return exact_path

    # 확장자가 다른 경우 stem 기준으로 검색
    candidates = [
        p for p in label_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and p.stem == image_path.stem
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        print(
            f"[경고] label 후보가 여러 개입니다: "
            f"{image_path.name}"
        )
        return candidates[0]

    return None


# ============================================================
# 3. 클래스별 resize
# ============================================================

report = []


for src_class, dst_class in CLASS_FOLDERS.items():

    # -------------------------
    # 원본 폴더
    # -------------------------

    src_image_dir = ROOT / src_class / "data"
    src_label_dir = ROOT / src_class / "label"

    # -------------------------
    # 새 폴더
    # -------------------------

    dst_image_dir = ROOT / dst_class / "image"
    dst_label_dir = ROOT / dst_class / "label"

    dst_image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dst_label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print(f"{src_class} -> {dst_class}")
    print("=" * 60)

    # -------------------------
    # 이미지 목록
    # -------------------------

    image_paths = sorted([
        p for p in src_image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
    ])

    print(
        f"원본 이미지 수: {len(image_paths)}"
    )

    success = 0
    failed = 0

    for idx, image_path in enumerate(
        image_paths,
        start=1
    ):

        # ==========================
        # label 찾기
        # ==========================

        label_path = find_label(
            src_label_dir,
            image_path
        )

        if label_path is None:

            print(
                f"[LABEL 없음] "
                f"{image_path.name}"
            )

            report.append({
                "class": src_class,
                "filename": image_path.name,
                "status": "label_not_found"
            })

            failed += 1
            continue

        try:

            # ==================================================
            # RGB image
            # ==================================================

            with Image.open(image_path) as img:

                img = img.convert("RGB")

                original_size = img.size

                resized_img = img.resize(
                    TARGET_SIZE,
                    Image.Resampling.LANCZOS
                )

                # JPEG 재압축 영향을 피하기 위해 PNG로 저장
                output_image_name = (
                    image_path.stem + ".png"
                )

                resized_img.save(
                    dst_image_dir /
                    output_image_name,
                    format="PNG"
                )

            # ==================================================
            # segmentation label
            # ==================================================

            with Image.open(label_path) as mask:

                # binary mask이므로
                # 보간값이 생기지 않도록 NEAREST
                resized_mask = mask.resize(
                    TARGET_SIZE,
                    Image.Resampling.NEAREST
                )

                output_label_name = (
                    image_path.stem + ".png"
                )

                resized_mask.save(
                    dst_label_dir /
                    output_label_name,
                    format="PNG"
                )

            success += 1

            report.append({
                "class": src_class,
                "filename": image_path.name,
                "original_width": original_size[0],
                "original_height": original_size[1],
                "new_width": 256,
                "new_height": 256,
                "status": "success"
            })

            print(
                f"[{idx}/{len(image_paths)}] "
                f"{image_path.name} -> 256×256"
            )

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] "
                f"{image_path.name}: {e}"
            )

            report.append({
                "class": src_class,
                "filename": image_path.name,
                "status": f"error: {e}"
            })

    print(
        f"\n완료: {success}"
    )

    print(
        f"실패: {failed}"
    )


# ============================================================
# 4. 처리 결과 저장
# ============================================================

report_df = pd.DataFrame(report)

report_path = (
    ROOT / "resize_256_report.csv"
)

report_df.to_csv(
    report_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. 최종 폴더 검사
# ============================================================

print("\n")
print("=" * 70)
print("최종 결과 확인")
print("=" * 70)

total_images = 0
total_labels = 0

for dst_class in CLASS_FOLDERS.values():

    image_dir = (
        ROOT / dst_class / "image"
    )

    label_dir = (
        ROOT / dst_class / "label"
    )

    image_files = list(
        image_dir.glob("*.png")
    )

    label_files = list(
        label_dir.glob("*.png")
    )

    print(
        f"{dst_class:15s} "
        f"image={len(image_files):3d} | "
        f"label={len(label_files):3d}"
    )

    total_images += len(image_files)
    total_labels += len(label_files)


print("-" * 70)

print(
    f"전체 image: {total_images}"
)

print(
    f"전체 label: {total_labels}"
)

print(
    f"결과 CSV: {report_path}"
)

print("=" * 70)