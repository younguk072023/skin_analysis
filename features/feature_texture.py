'''
Texture Feature Groups
'''

feature_texture = {

    # ========================================================
    # 1. Lesion texture only
    # ========================================================

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


    # ========================================================
    # 2. Background texture only
    # ========================================================

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


    # ========================================================
    # 3. Texture difference only
    # lesion - background
    # ========================================================

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


    # ========================================================
    # 4. Lesion texture group
    # ========================================================

    "lesion_texture_group": [
        "lesion_glcm_contrast",
        "lesion_glcm_homogeneity",
        "lesion_glcm_energy",
        "lesion_glcm_correlation",
        "lesion_glcm_entropy",
        "lesion_lbp_entropy",
    ],


    # ========================================================
    # 5. Background texture group
    # ========================================================

    "background_texture_group": [
        "bg_glcm_contrast",
        "bg_glcm_homogeneity",
        "bg_glcm_energy",
        "bg_glcm_correlation",
        "bg_glcm_entropy",
        "bg_lbp_entropy",
    ],


    # ========================================================
    # 6. Delta texture group
    # lesion - background
    # ========================================================

    "delta_texture_group": [
        "delta_glcm_contrast",
        "delta_glcm_homogeneity",
        "delta_glcm_energy",
        "delta_glcm_correlation",
        "delta_glcm_entropy",
        "delta_lbp_entropy",
    ],


    # ========================================================
    # 7. GLCM only
    # ========================================================

    "glcm_group": [

        # Lesion
        "lesion_glcm_contrast",
        "lesion_glcm_homogeneity",
        "lesion_glcm_energy",
        "lesion_glcm_correlation",
        "lesion_glcm_entropy",

        # Background
        "bg_glcm_contrast",
        "bg_glcm_homogeneity",
        "bg_glcm_energy",
        "bg_glcm_correlation",
        "bg_glcm_entropy",

        # Lesion - Background
        "delta_glcm_contrast",
        "delta_glcm_homogeneity",
        "delta_glcm_energy",
        "delta_glcm_correlation",
        "delta_glcm_entropy",
    ],


    # ========================================================
    # 8. LBP only
    # ========================================================

    "lbp_group": [
        "lesion_lbp_entropy",
        "bg_lbp_entropy",
        "delta_lbp_entropy",
    ],


    # ========================================================
    # 9. Raw texture
    # lesion + background
    # ========================================================

    "raw_texture_group": [

        # Lesion
        "lesion_glcm_contrast",
        "lesion_glcm_homogeneity",
        "lesion_glcm_energy",
        "lesion_glcm_correlation",
        "lesion_glcm_entropy",
        "lesion_lbp_entropy",

        # Background
        "bg_glcm_contrast",
        "bg_glcm_homogeneity",
        "bg_glcm_energy",
        "bg_glcm_correlation",
        "bg_glcm_entropy",
        "bg_lbp_entropy",
    ],


    # ========================================================
    # 10. Texture difference
    # delta only
    # ========================================================

    "texture_difference_group": [
        "delta_glcm_contrast",
        "delta_glcm_homogeneity",
        "delta_glcm_energy",
        "delta_glcm_correlation",
        "delta_glcm_entropy",
        "delta_lbp_entropy",
    ],


    # ========================================================
    # 11. All texture features
    # ========================================================

    "all_texture_group": [

        # Lesion
        "lesion_glcm_contrast",
        "lesion_glcm_homogeneity",
        "lesion_glcm_energy",
        "lesion_glcm_correlation",
        "lesion_glcm_entropy",
        "lesion_lbp_entropy",

        # Background
        "bg_glcm_contrast",
        "bg_glcm_homogeneity",
        "bg_glcm_energy",
        "bg_glcm_correlation",
        "bg_glcm_entropy",
        "bg_lbp_entropy",

        # Lesion - Background
        "delta_glcm_contrast",
        "delta_glcm_homogeneity",
        "delta_glcm_energy",
        "delta_glcm_correlation",
        "delta_glcm_entropy",
        "delta_lbp_entropy",
    ],
}