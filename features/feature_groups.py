feature_groups = {

    # =========================
    # Lesion raw value only
    # =========================
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

    # =========================
    # Background raw value only
    # =========================
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

    # =========================
    # Mean difference only
    # =========================
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

    # =========================
    # Absolute difference only
    # =========================
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

    # =========================
    # Wasserstein distance only
    # =========================
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

    # =========================
    # Grouped feature sets
    # =========================
    "lesion_group": [
        "lesion_L_mean",
        "lesion_a_mean",
        "lesion_b_mean",
        "lesion_ITA_mean",
    ],

    "background_group": [
        "bg_L_mean",
        "bg_a_mean",
        "bg_b_mean",
        "bg_ITA_mean",
    ],
    

    "delta_group": [    
        "delta_L_mean",
        "delta_a_mean",
        "delta_b_mean",
        "delta_ITA_mean",
    ],

    "abs_delta_group": [
        "abs_delta_L_mean",
        "abs_delta_a_mean",
        "abs_delta_b_mean",
        "abs_delta_ITA_mean",
    ],

    "wasserstein_group": [
        "wasserstein_L",
        "wasserstein_a",
        "wasserstein_b",
        "wasserstein_ITA",
    ],
    
    # =========================
    #  각 l,a,b등 값에 대한 feature
    # =========================

    "L_group": [
        "lesion_L_mean",
        "bg_L_mean",
        "delta_L_mean",
        "abs_delta_L_mean",
        "wasserstein_L",
    ],

    "a_group": [
        "lesion_a_mean",
        "bg_a_mean",
        "delta_a_mean",
        "abs_delta_a_mean",
        "wasserstein_a",
    ],

    "b_group": [
        "lesion_b_mean",
        "bg_b_mean",
        "delta_b_mean",
        "abs_delta_b_mean",
        "wasserstein_b",
    ],

    "ITA_group": [
        "lesion_ITA_mean",
        "bg_ITA_mean",
        "delta_ITA_mean",
        "abs_delta_ITA_mean",
        "wasserstein_ITA",
    ],

    # =========================
    #  조합 feature
    # =========================

    # 병변 + 배경의 원시 평균값
    "raw_group": [
        "lesion_L_mean",
        "lesion_a_mean",
        "lesion_b_mean",
        "lesion_ITA_mean",

        "bg_L_mean",
        "bg_a_mean",
        "bg_b_mean",
        "bg_ITA_mean",
    ],

    # 방향성 차이 + 절대 차이
    "mean_difference_group": [
        "delta_L_mean",
        "delta_a_mean",
        "delta_b_mean",
        "delta_ITA_mean",

        "abs_delta_L_mean",
        "abs_delta_a_mean",
        "abs_delta_b_mean",
        "abs_delta_ITA_mean",
    ],

    # 모든 대비 특징
    "contrast_group": [
        "delta_L_mean",
        "delta_a_mean",
        "delta_b_mean",
        "delta_ITA_mean",

        "abs_delta_L_mean",
        "abs_delta_a_mean",
        "abs_delta_b_mean",
        "abs_delta_ITA_mean",

        "wasserstein_L",
        "wasserstein_a",
        "wasserstein_b",
        "wasserstein_ITA",
    ],

    # 병변값 + 모든 대비
    "lesion_contrast_group": [
        "lesion_L_mean",
        "lesion_a_mean",
        "lesion_b_mean",
        "lesion_ITA_mean",

        "delta_L_mean",
        "delta_a_mean",
        "delta_b_mean",
        "delta_ITA_mean",

        "abs_delta_L_mean",
        "abs_delta_a_mean",
        "abs_delta_b_mean",
        "abs_delta_ITA_mean",

        "wasserstein_L",
        "wasserstein_a",
        "wasserstein_b",
        "wasserstein_ITA",
    ],

    # 배경값 + 모든 대비
    "background_contrast_group": [
        "bg_L_mean",
        "bg_a_mean",
        "bg_b_mean",
        "bg_ITA_mean",

        "delta_L_mean",
        "delta_a_mean",
        "delta_b_mean",
        "delta_ITA_mean",

        "abs_delta_L_mean",
        "abs_delta_a_mean",
        "abs_delta_b_mean",
        "abs_delta_ITA_mean",

        "wasserstein_L",
        "wasserstein_a",
        "wasserstein_b",
        "wasserstein_ITA",
    ],
        

    #모든 전체 특징
    "all_features_group": [
        "lesion_L_mean",
        "lesion_a_mean",
        "lesion_b_mean",
        "lesion_ITA_mean",

        "bg_L_mean",
        "bg_a_mean",
        "bg_b_mean",
        "bg_ITA_mean",

        "delta_L_mean",
        "delta_a_mean",
        "delta_b_mean",
        "delta_ITA_mean",

        "abs_delta_L_mean",
        "abs_delta_a_mean",
        "abs_delta_b_mean",
        "abs_delta_ITA_mean",

        "wasserstein_L",
        "wasserstein_a",
        "wasserstein_b",
        "wasserstein_ITA",
    ],
}