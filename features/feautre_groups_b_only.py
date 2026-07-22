'''
통계적으로 높았던 lesion_b를 뺀 features
'''


feature_groups = {

    # =========================
    # Grouped feature sets
    # =========================
    "lesion_group": [
        "lesion_L_mean",
        "lesion_a_mean",
        "lesion_ITA_mean",
    ],

    "b_group": [

        "bg_b_mean",
        "delta_b_mean",
        "abs_delta_b_mean",
        "wasserstein_b",
    ],

    # =========================
    #  조합 feature
    # =========================

    # 병변 + 배경의 원시 평균값
    "raw_group": [
        "lesion_L_mean",
        "lesion_a_mean",
        "lesion_ITA_mean",

        "bg_L_mean",
        "bg_a_mean",
        "bg_b_mean",
        "bg_ITA_mean",
    ],


    # 병변값 + 모든 대비
    "lesion_contrast_group": [
        "lesion_L_mean",
        "lesion_a_mean",
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
        
    #모든 전체 특징
    "all_features_group": [
        "lesion_L_mean",
        "lesion_a_mean",
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