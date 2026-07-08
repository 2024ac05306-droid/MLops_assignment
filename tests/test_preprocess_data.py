import pandas as pd

from preprocess_data import TARGET_COLUMN, PreprocessData


def raw_heart_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
            [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
            [37, 0, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, None, 7, 1],
            [41, 0, 2, 120, 204, 0, 0, 172, 0, 1.4, 1, 0, None, 0],
        ],
        columns=PreprocessData().column_names,
    )


def test_load_data_converts_values_and_binarizes_target(tmp_path):
    input_file = tmp_path / "heart.csv"
    input_file.write_text(
        "\n".join(
            [
                "63,1,1,145,233,1,2,150,0,2.3,3,0,6,0",
                "67,1,4,160,286,0,2,108,1,1.5,2,3,3,2",
                "37,0,3,130,250,0,0,187,0,3.5,3,?,7,1",
            ]
        ),
        encoding="utf-8",
    )

    data = PreprocessData().load_data(str(input_file))

    assert list(data.columns) == PreprocessData().column_names
    assert data["ca"].isna().sum() == 1
    assert data[TARGET_COLUMN].tolist() == [0, 1, 1]


def test_fit_transform_returns_numeric_features_and_target():
    processor = PreprocessData()
    processed = processor.fit_transform(raw_heart_data())

    assert TARGET_COLUMN in processed.columns
    assert processed.shape[0] == 4
    assert processed.drop(columns=[TARGET_COLUMN]).isna().sum().sum() == 0
    assert set(processed[TARGET_COLUMN].unique()) == {0, 1, 2}
    assert any(column.startswith("numeric__age") for column in processed.columns)
    assert any(column.startswith("categorical__cp") for column in processed.columns)


def test_transform_uses_fitted_preprocessor_for_unseen_category():
    processor = PreprocessData()
    processor.fit_transform(raw_heart_data())

    inference_data = raw_heart_data().drop(columns=[TARGET_COLUMN]).iloc[[0]].copy()
    inference_data.loc[:, "thal"] = 99

    transformed = processor.transform(inference_data)

    assert transformed.shape[0] == 1
    assert TARGET_COLUMN not in transformed.columns
    assert transformed.isna().sum().sum() == 0


def test_save_processed_data_and_preprocessor_create_files(tmp_path):
    processor = PreprocessData()
    processed = processor.fit_transform(raw_heart_data())

    data_path = tmp_path / "processed" / "heart.csv"
    preprocessor_path = tmp_path / "models" / "preprocessor.pkl"

    processor.save_processed_data(processed, str(data_path))
    processor.save_preprocessor(str(preprocessor_path))

    assert data_path.exists()
    assert preprocessor_path.exists()
    assert pd.read_csv(data_path).shape == processed.shape
