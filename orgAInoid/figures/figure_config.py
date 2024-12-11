AXIS_LABEL_SIZE = 6
TITLE_SIZE = 8
UMAP_LABEL_SIZE = 6

DPI = 300

EXPERIMENTS = [
    "E001",
    "E002",
    "E004",
    "E005",
    "E006",
    "E007",
    "E008",
    "E009",
    "E010",
    "E011",
    "E012",
    #"E013",
    #"E014",
    #"E017",
    #"E019",
    #"E020"
]

EXPERIMENT_MAP = {
    "E001": "E001",
    "E002": "E002",
    "E004": "E003",
    "E005": "E004",
    "E006": "E005",
    "E007": "E006",
    "E008": "E007",
    "E009": "E008",
    "E010": "E009",
    "E011": "E010",
    "E012": "E011",
    "E013": "E015",
    "E014": "E012",
    "E017": "E016",
    "E019": "E013",
    "E020": "E014"
}

RPE_CUTOFFS = [956.9, 1590.4]
LENS_CUTOFFS = [16324.85763, 29083.23]

RPE_UM_CONVERSION_FACTOR = 2.1786**2

FIGURE_WIDTH_FULL = 6.75
FIGURE_WIDTH_HALF = FIGURE_WIDTH_FULL / 2

FIGURE_HEIGHT_FULL = 9.375
FIGURE_HEIGHT_HALF = FIGURE_HEIGHT_FULL / 2

SUPERVISED_SCORE = "f1_score"
UNSUPERVISED_SCORE = "jaccard_score"

#SCORING_YLIMS = (-0.25 , 1.25)
SCORING_YLIMS = (-0.15, 1.05)

TRAIN_SIZES = ["50", "500", "5000", "50000", "500000"]

SUPERVISED_UMAP_PALETTE = "Set1"

CONF_MATRIX_COLORS = ["#31688E", "#35B779", "#FDE725", "#440154"]

CONF_MATRIX_LABEL_DICT = {"fp": "false pos.", "fn": "false neg.", "tp": "true pos.", "tn": "true neg."}

EXPERIMENT_LEGEND_CMAP = "tab20"

TWO_COL_LEGEND = {
    "ncol": 2,                    # Two columns
    "columnspacing": 0.1,         # Reduce spacing between columns
    "handletextpad": 0.1,         # Reduce spacing between handles and text
    "borderaxespad": 0.1          # Reduce padding around the legend box
}

STRIPPLOT_PARAMS = {
    "linewidth": 0.5,
    "dodge": True,
    "s": 2
}

BOXPLOT_PARAMS = {
    "boxprops": dict(facecolor = "white"),
    "whis": (0,100),
    "linewidth": 1,
    "showfliers": False
}

XTICKLABEL_PARAMS = {
    "ha": "right",
    "rotation": 45,
    "rotation_mode": "anchor",
    "fontsize": AXIS_LABEL_SIZE
}

TICKPARAMS_PARAMS = {
    "axis": "both",
    "labelsize": AXIS_LABEL_SIZE
}

CENTERED_LEGEND_PARAMS = {
    "bbox_to_anchor": (1, 0.5),
    "loc": "center left",
    "fontsize": AXIS_LABEL_SIZE,
    "markerscale": 0.5
}
