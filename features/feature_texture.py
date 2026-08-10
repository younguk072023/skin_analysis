'''
texture feature

'''

feature_texture = {

    # =========================
    # Lesion texture only
    # =========================
    "lesion_contrast_only": [
        "lesion_glcm_contrast",
    ],

    "lesion_homogeneity_only": [
        "lesion_glcm_homogeneity",
    ],

    "lesion_energy_only": [
        "lesion_glcm_energy",
    ],

    "lesion_correlation_only": [
        "lesion_glcm_correlation",
    ],

    "lesion_entropy_only": [
        "lesion_glcm_entropy",
    ],

    "lesion_lbp_entropy_only": [
        "lesion_lbp_entropy",
    ],


    # =========================
    # Background texture only
    # =========================
    "bg_contrast_only": [
        "bg_glcm_contrast",
    ],

    "bg_homogeneity_only": [
        "bg_glcm_homogeneity",
    ],

    "bg_energy_only": [
        "bg_glcm_energy",
    ],

    "bg_correlation_only": [
        "bg_glcm_correlation",
    ],

    "bg_entropy_only": [
        "bg_glcm_entropy",
    ],

    "bg_lbp_entropy_only": [
        "bg_lbp_entropy",
    ],


    # =========================
    # Texture difference only
    # lesion - background
    # =========================
    "delta_contrast_only": [
        "delta_glcm_contrast",
    ],

    "delta_homogeneity_only": [
        "delta_glcm_homogeneity",
    ],

    "delta_energy_only": [
        "delta_glcm_energy",
    ],

    "delta_correlation_only": [
        "delta_glcm_correlation",
    ],

    "delta_entropy_only": [
        "delta_glcm_entropy",
    ],

    "delta_lbp_entropy_only": [
        "delta_lbp_entropy",
    ],


    # =========================
    # Absolute texture difference
    # =========================
    "abs_delta_contrast_only": [
        "abs_delta_glcm_contrast",
    ],

    "abs_delta_homogeneity_only": [
        "abs_delta_glcm_homogeneity",
    ],

    "abs_delta_energy_only": [
        "abs_delta_glcm_energy",
    ],

    "abs_delta_correlation_only": [
        "abs_delta_glcm_correlation",
    ],

    "abs_delta_entropy_only": [
        "abs_delta_glcm_entropy",
    ],

    "abs_delta_lbp_entropy_only": [
        "abs_delta_lbp_entropy",
    ],


    # =========================
    # Grouped texture sets
    # =========================

    # 병변 texture
    "lesion_texture_group": [
        "lesion_glcm_contrast",
        "lesion_glcm_homogeneity",
        "lesion_glcm_energy",
        "lesion_glcm_correlation",
        "lesion_glcm_entropy",
        "lesion_lbp_entropy",
    ],

    # 배경 texture
    "background_texture_group": [
        "bg_glcm_contrast",
        "bg_glcm_homogeneity",
        "bg_glcm_energy",
        "bg_glcm_correlation",
        "bg_glcm_entropy",
        "bg_lbp_entropy",
    ],

    # 방향성 texture 차이
    "delta_texture_group": [
        "delta_glcm_contrast",
        "delta_glcm_homogeneity",
        "delta_glcm_energy",
        "delta_glcm_correlation",
        "delta_glcm_entropy",
        "delta_lbp_entropy",
    ],

    # texture 차이 크기
    "abs_delta_texture_group": [
        "abs_delta_glcm_contrast",
        "abs_delta_glcm_homogeneity",
        "abs_delta_glcm_energy",
        "abs_delta_glcm_correlation",
        "abs_delta_glcm_entropy",
        "abs_delta_lbp_entropy",
    ],


    # =========================
    # GLCM only
    # =========================
    "glcm_group": [
        "lesion_glcm_contrast",
        "lesion_glcm_homogeneity",
        "lesion_glcm_energy",
        "lesion_glcm_correlation",
        "lesion_glcm_entropy",

        "bg_glcm_contrast",
        "bg_glcm_homogeneity",
        "bg_glcm_energy",
        "bg_glcm_correlation",
        "bg_glcm_entropy",

        "delta_glcm_contrast",
        "delta_glcm_homogeneity",
        "delta_glcm_energy",
        "delta_glcm_correlation",
        "delta_glcm_entropy",

        "abs_delta_glcm_contrast",
        "abs_delta_glcm_homogeneity",
        "abs_delta_glcm_energy",
        "abs_delta_glcm_correlation",
        "abs_delta_glcm_entropy",
    ],


    # =========================
    # LBP only
    # =========================
    "lbp_group": [
        "lesion_lbp_entropy",
        "bg_lbp_entropy",
        "delta_lbp_entropy",
        "abs_delta_lbp_entropy",
    ],


    # =========================
    # Raw texture
    # lesion + background
    # =========================
    "raw_texture_group": [
        "lesion_glcm_contrast",
        "lesion_glcm_homogeneity",
        "lesion_glcm_energy",
        "lesion_glcm_correlation",
        "lesion_glcm_entropy",
        "lesion_lbp_entropy",

        "bg_glcm_contrast",
        "bg_glcm_homogeneity",
        "bg_glcm_energy",
        "bg_glcm_correlation",
        "bg_glcm_entropy",
        "bg_lbp_entropy",
    ],


    # =========================
    # Texture contrast
    # delta + abs delta
    # =========================
    "texture_contrast_group": [
        "delta_glcm_contrast",
        "delta_glcm_homogeneity",
        "delta_glcm_energy",
        "delta_glcm_correlation",
        "delta_glcm_entropy",
        "delta_lbp_entropy",

        "abs_delta_glcm_contrast",
        "abs_delta_glcm_homogeneity",
        "abs_delta_glcm_energy",
        "abs_delta_glcm_correlation",
        "abs_delta_glcm_entropy",
        "abs_delta_lbp_entropy",
    ],


    # =========================
    # All texture features
    # =========================
    "all_texture_group": [
        "lesion_glcm_contrast",
        "lesion_glcm_homogeneity",
        "lesion_glcm_energy",
        "lesion_glcm_correlation",
        "lesion_glcm_entropy",
        "lesion_lbp_entropy",

        "bg_glcm_contrast",
        "bg_glcm_homogeneity",
        "bg_glcm_energy",
        "bg_glcm_correlation",
        "bg_glcm_entropy",
        "bg_lbp_entropy",

        "delta_glcm_contrast",
        "delta_glcm_homogeneity",
        "delta_glcm_energy",
        "delta_glcm_correlation",
        "delta_glcm_entropy",
        "delta_lbp_entropy",

        "abs_delta_glcm_contrast",
        "abs_delta_glcm_homogeneity",
        "abs_delta_glcm_energy",
        "abs_delta_glcm_correlation",
        "abs_delta_glcm_entropy",
        "abs_delta_lbp_entropy",
    ],
}