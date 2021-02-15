import os

from utils import process_config


model_key = {
    "P": "piece",
    "N": "piece",
    "B": "piece",
    "R": "piece",
    "Q": "piece",
    "K": "piece",
    "picker": "piece",
}


def init_models(model_key):
    models = {}
    for k, v in model_key.items():
        module = __import__("models." + v + "_model", fromlist=[v + "_model"])
        model = getattr(module, v.capitalize())
        models[k] = model
    return models


def init_trainers(model_key):
    trainers = {}
    for k, v in model_key.items():
        module = __import__("trainers." + v + "_trainer", fromlist=[v + "_trainer"])
        trainer = getattr(module, v.capitalize() + "Trainer")
        trainers[k] = trainer
    return trainers


def get_data_files(model_type, t_or_v):
    files = []
    for file_name in os.listdir(f"./data/parsed/{t_or_v}/"):
        if file_name.split("_")[0] == model_type:
            files.append(f"./data/parsed/{t_or_v}/" + file_name)
    return files


def parse_fileobjs(file_names):
    parsed_files = []
    for filename in file_names:
        model, data_type, _, series = filename.split("_")
        parsed_files.append(
            {
                "filename": filename,
                "series": series,
                "type": data_type,
            }
        )
    return parsed_files


def sort_files(fileobj):
    return (fileobj["series"], fileobj["type"])


if __name__ == "__main__":
    models = init_models(model_key)
    trainers = init_trainers(model_key)

    for model_type, model_instance in models.items():
        config_file = "./src/configs/" + model_key[model_type] + ".json"
        config = process_config(config_file)
        model = model_instance(config)

        trainer = trainers[model_type](model, config)
        trainer.prep_train()

        train_files = get_data_files(model_type, "train")
        parsed_fileobjs = parse_fileobjs(train_files)
        sorted_files = sorted(parsed_fileobjs, key=sort_files)
        sorted_files = [file["filename"] for file in sorted_files]
        train_file_pairs = zip(sorted_files[::2], sorted_files[1::2])

        valid_files = get_data_files(model_type, "validation")
        parsed_fileobjs = parse_fileobjs(valid_files)
        sorted_files = sorted(parsed_fileobjs, key=sort_files)
        sorted_files = [file["filename"] for file in sorted_files]
        valid_file_pairs = zip(sorted_files[::2], sorted_files[1::2])

        trainer.train(train_file_pairs, valid_file_pairs)

        model.save()
