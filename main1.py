import logging
from ludwig.api import LudwigModel
from ludwig.datasets import agnews

# Loads the dataset as a pandas.DataFrame
train_df, test_df, _ = agnews.load(split=True)

# Prints a preview of the first five rows.
train_df.head(5)




config = {
    "input_features": [
        {
            "name": "title",  # The name of the input column
            "type": "text",  # Data type of the input column
            "encoder": {
                "type": "auto_transformer",
                "pretrained_model_name_or_path": "bigscience/bloom-560m",
                "trainable": True,
                "reduce_output": "sum",
                "pretrained_kwargs": None,
            },
        },
    ],
    "output_features": [
        {
            "name": "class",
            "type": "category",
        }
    ],
    "trainer": {
        "learning_rate": 0.00001,
        "epochs": 3,  # We'll train for three epochs. Training longer might give
        "batch_size": 256,
    },
    "backend": {
        "type": "ray",
        "cache_dir":'s3://ludwigconfig/ludwig_config/',
        "cache_credentials": '/home/ray/.aws/credentials.json',
        "processor": {
            "type": "dask",
            "parallelism": 16,
            "persist": True,
        },
        "trainer": {
            "strategy": "fsdp",
            # "use_gpu": True,
            # "num_workers": 2,
        # "loader": {
        #     "fully_executed": False
        #   }
        }
    }
}



model = LudwigModel(config, logging_level=logging.INFO)
train_stats, preprocessed_data, output_directory = model.train(dataset=train_df,)