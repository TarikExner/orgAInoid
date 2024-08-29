import os
import pandas as pd

def _generate_file_table(experiment_id: str,
                         image_dir: str,
                         annotations_file: str):

    annotation_table = pd.read_csv(annotations_file)

    annotation_table["well"] = [
        entry.split(experiment_id)[1]
        if not entry == f"{experiment_id}{experiment_id}" else experiment_id
        for entry in annotation_table["ID"].tolist()
    ]
    annotation_table["experiment"] = experiment_id

    annotations = [
        col for col in annotation_table if col not in ["ID", "well"]
    ]

    file_name_information = [
        "experiment", "well", "file_name", "position", "slice", "loop"
    ]

    metadata_dict = {
        annotation: []
        for annotation in file_name_information + annotations
    }

    files = os.listdir(image_dir)
    files = [file for file in files if file.endswith(".tif")]
    for file_name in files:
        contents = file_name.split("-")
        contents = [entry for entry in contents if entry != ""]
        if len(contents) != 14:
            print(f"Invalid image: {file_name}")
            continue
        well = contents[0]
        metadata_dict["experiment"].append(experiment_id)
        metadata_dict["file_name"].append(file_name)
        metadata_dict["well"].append(well)
        metadata_dict["position"].append(contents[1])
        metadata_dict["loop"].append(contents[2])
        metadata_dict["slice"].append(contents[4])
        metadata_dict["file_name"].append(file_name)
        metadata = annotation_table.loc[
            (annotation_table["well"] == well) &
            (annotation_table["experiment"] == experiment_id),
            annotations
        ]
        for annotation in annotations:
            metadata_dict[annotation].append(metadata[annotation].iloc[0])

    return pd.DataFrame(metadata_dict)
