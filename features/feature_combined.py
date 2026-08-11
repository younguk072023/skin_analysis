from features.feature_color import feature_color
from features.feature_texture import feature_texture


'''
Color + Texture Feature Groups
'''

feature_combined = {

    # ========================================================
    # 1. Lesion Color + Lesion Texture
    #
    # 병변 자체의 색상 + 질감
    # ========================================================

    "lesion_color_texture_group": (
        feature_color[
            "lesion_color_group"
        ]
        +
        feature_texture[
            "lesion_texture_group"
        ]
    ),


    # ========================================================
    # 2. Background Color + Background Texture
    #
    # 주변 피부 자체의 색상 + 질감
    # background가 없는 영상은 제외
    # ========================================================

    "background_color_texture_group": (
        feature_color[
            "background_color_group"
        ]
        +
        feature_texture[
            "background_texture_group"
        ]
    ),


    # ========================================================
    # 3. Delta Color + Delta Texture
    #
    # 병변 - 주변 피부의
    # 색상 차이 + 질감 차이
    # ========================================================

    "delta_color_texture_group": (
        feature_color[
            "delta_color_group"
        ]
        +
        feature_texture[
            "delta_texture_group"
        ]
    ),


    # ========================================================
    # 4. Raw Color + Raw Texture
    #
    # 병변과 배경의 직접 측정값
    # ========================================================

    "raw_color_texture_group": (
        feature_color[
            "raw_color_group"
        ]
        +
        feature_texture[
            "raw_texture_group"
        ]
    ),


    # ========================================================
    # 5. Lesion-Background Contrast
    #
    # Color:
    #   delta
    #   abs_delta
    #   Wasserstein
    #
    # Texture:
    #   delta
    #
    # 병변과 주변 피부의 상대적 차이에 초점
    # ========================================================

    "color_texture_contrast_group": (
        feature_color[
            "color_contrast_group"
        ]
        +
        feature_texture[
            "texture_difference_group"
        ]
    ),


    # ========================================================
    # 6. All Color + All Texture
    #
    # Color   = 20
    # Texture = 18
    # Total   = 38
    # ========================================================

    "all_color_texture_group": (
        feature_color[
            "all_color_group"
        ]
        +
        feature_texture[
            "all_texture_group"
        ]
    ),
}