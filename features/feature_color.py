'''
Color Feature Groups
'''

feature_color = {

    # ========================================================
    # 1. Lesion color only
    # ========================================================

    "lesion_L_only": [
        "lesion_L_mean",
    ],

    "lesion_a_only": [
        "lesion_a_mean",
    ],

    "lesion_b_only": [
        "lesion_b_mean",
    ],

    "lesion_ITA_only": [
        "lesion_ITA_mean",
    ],


    # ========================================================
    # 2. Background color only
    # ========================================================

    "bg_L_only": [
        "bg_L_mean",
    ],

    "bg_a_only": [
        "bg_a_mean",
    ],

    "bg_b_only": [
        "bg_b_mean",
    ],

    "bg_ITA_only": [
        "bg_ITA_mean",
    ],


    # ========================================================
    # 3. Directional color difference only
    # lesion - background
    # ========================================================

    "delta_L_only": [
        "delta_L_mean",
    ],

    "delta_a_only": [
        "delta_a_mean",
    ],

    "delta_b_only": [
        "delta_b_mean",
    ],

    "delta_ITA_only": [
        "delta_ITA_mean",
    ],


    # ========================================================
    # 4. Absolute color difference only
    # |lesion - background|
    # ========================================================

    "abs_delta_L_only": [
        "abs_delta_L_mean",
    ],

    "abs_delta_a_only": [
        "abs_delta_a_mean",
    ],

    "abs_delta_b_only": [
        "abs_delta_b_mean",
    ],

    "abs_delta_ITA_only": [
        "abs_delta_ITA_mean",
    ],


    # ========================================================
    # 5. Wasserstein distance only
    # ========================================================

    "wd_L_only": [
        "wasserstein_L",
    ],

    "wd_a_only": [
        "wasserstein_a",
    ],

    "wd_b_only": [
        "wasserstein_b",
    ],

    "wd_ITA_only": [
        "wasserstein_ITA",
    ],


    # ========================================================
    # 6. Lesion color group
    # ========================================================

    "lesion_color_group": [
        "lesion_L_mean",
        "lesion_a_mean",
        "lesion_b_mean",
        "lesion_ITA_mean",
    ],


    # ========================================================
    # 7. Background color group
    # ========================================================

    "background_color_group": [
        "bg_L_mean",
        "bg_a_mean",
        "bg_b_mean",
        "bg_ITA_mean",
    ],


    # ========================================================
    # 8. Directional color difference group
    # lesion - background
    # ========================================================

    "delta_color_group": [
        "delta_L_mean",
        "delta_a_mean",
        "delta_b_mean",
        "delta_ITA_mean",
    ],


    # ========================================================
    # 9. Absolute color difference group
    # ========================================================

    "abs_delta_color_group": [
        "abs_delta_L_mean",
        "abs_delta_a_mean",
        "abs_delta_b_mean",
        "abs_delta_ITA_mean",
    ],


    # ========================================================
    # 10. Wasserstein color group
    # ========================================================

    "wasserstein_color_group": [
        "wasserstein_L",
        "wasserstein_a",
        "wasserstein_b",
        "wasserstein_ITA",
    ],


    # ========================================================
    # 11. L* color features
    # ========================================================

    "L_color_group": [
        "lesion_L_mean",
        "bg_L_mean",
        "delta_L_mean",
        "abs_delta_L_mean",
        "wasserstein_L",
    ],


    # ========================================================
    # 12. a* color features
    # ========================================================

    "a_color_group": [
        "lesion_a_mean",
        "bg_a_mean",
        "delta_a_mean",
        "abs_delta_a_mean",
        "wasserstein_a",
    ],


    # ========================================================
    # 13. b* color features
    # ========================================================

    "b_color_group": [
        "lesion_b_mean",
        "bg_b_mean",
        "delta_b_mean",
        "abs_delta_b_mean",
        "wasserstein_b",
    ],


    # ========================================================
    # 14. ITA color features
    # ========================================================

    "ITA_color_group": [
        "lesion_ITA_mean",
        "bg_ITA_mean",
        "delta_ITA_mean",
        "abs_delta_ITA_mean",
        "wasserstein_ITA",
    ],


    # ========================================================
    # 15. Raw color group
    # lesion + background
    # ========================================================

    "raw_color_group": [

        # Lesion
        "lesion_L_mean",
        "lesion_a_mean",
        "lesion_b_mean",
        "lesion_ITA_mean",

        # Background
        "bg_L_mean",
        "bg_a_mean",
        "bg_b_mean",
        "bg_ITA_mean",
    ],


    # ========================================================
    # 16. Color difference group
    # delta + absolute delta
    # ========================================================

    "color_difference_group": [

        # Directional difference
        "delta_L_mean",
        "delta_a_mean",
        "delta_b_mean",
        "delta_ITA_mean",

        # Absolute difference
        "abs_delta_L_mean",
        "abs_delta_a_mean",
        "abs_delta_b_mean",
        "abs_delta_ITA_mean",
    ],


    # ========================================================
    # 17. Color contrast group
    # delta + abs_delta + Wasserstein
    # ========================================================

    "color_contrast_group": [

        # Directional difference
        "delta_L_mean",
        "delta_a_mean",
        "delta_b_mean",
        "delta_ITA_mean",

        # Absolute difference
        "abs_delta_L_mean",
        "abs_delta_a_mean",
        "abs_delta_b_mean",
        "abs_delta_ITA_mean",

        # Distribution difference
        "wasserstein_L",
        "wasserstein_a",
        "wasserstein_b",
        "wasserstein_ITA",
    ],


    # ========================================================
    # 18. Lesion color + color contrast
    # ========================================================

    "lesion_color_contrast_group": [

        # Lesion
        "lesion_L_mean",
        "lesion_a_mean",
        "lesion_b_mean",
        "lesion_ITA_mean",

        # Directional difference
        "delta_L_mean",
        "delta_a_mean",
        "delta_b_mean",
        "delta_ITA_mean",

        # Absolute difference
        "abs_delta_L_mean",
        "abs_delta_a_mean",
        "abs_delta_b_mean",
        "abs_delta_ITA_mean",

        # Distribution difference
        "wasserstein_L",
        "wasserstein_a",
        "wasserstein_b",
        "wasserstein_ITA",
    ],


    # ========================================================
    # 19. Background color + color contrast
    # ========================================================

    "background_color_contrast_group": [

        # Background
        "bg_L_mean",
        "bg_a_mean",
        "bg_b_mean",
        "bg_ITA_mean",

        # Directional difference
        "delta_L_mean",
        "delta_a_mean",
        "delta_b_mean",
        "delta_ITA_mean",

        # Absolute difference
        "abs_delta_L_mean",
        "abs_delta_a_mean",
        "abs_delta_b_mean",
        "abs_delta_ITA_mean",

        # Distribution difference
        "wasserstein_L",
        "wasserstein_a",
        "wasserstein_b",
        "wasserstein_ITA",
    ],


    # ========================================================
    # 20. All color features
    # 총 20개
    # ========================================================

    "all_color_group": [

        # Lesion
        "lesion_L_mean",
        "lesion_a_mean",
        "lesion_b_mean",
        "lesion_ITA_mean",

        # Background
        "bg_L_mean",
        "bg_a_mean",
        "bg_b_mean",
        "bg_ITA_mean",

        # Directional difference
        "delta_L_mean",
        "delta_a_mean",
        "delta_b_mean",
        "delta_ITA_mean",

        # Absolute difference
        "abs_delta_L_mean",
        "abs_delta_a_mean",
        "abs_delta_b_mean",
        "abs_delta_ITA_mean",

        # Wasserstein distance
        "wasserstein_L",
        "wasserstein_a",
        "wasserstein_b",
        "wasserstein_ITA",
    ],
}