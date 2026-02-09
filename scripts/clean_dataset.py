"""..."""

from scripts.build_dataset import load_json, save_json, METADATA_FILE, OUTPUT_DIR


def sync_metadata():

    metadata = load_json(METADATA_FILE)

    keys_to_delete = []
    for image_guid in metadata.keys():
        expected_file_path = OUTPUT_DIR / f"{image_guid}.png"
        if not expected_file_path.exists():
            keys_to_delete.append(image_guid)

    for key in keys_to_delete:
        del metadata[key]

    save_json(metadata, METADATA_FILE)


if __name__ == "__main__":
    sync_metadata()
