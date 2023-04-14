import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import logging
from ludwig.api import LudwigModel
import ray


from ludwig.datasets import agnews

# Loads the dataset as a pandas.DataFrame
train_df, test_df, _ = agnews.load(split=True)
train_df.to_csv("new.csv")


# Prints a preview of the first five rows.
train_df.head(5)

import logging
from ludwig.api import LudwigModel


config = {
  "input_features": [
    {
      "name": "title",    # The name of the input column
      "type": "text",     # Data type of the input column
      "encoder": {
          "type": "auto_transformer",   # The model architecture to use
          "pretrained_model_name_or_path": "bigscience/bloom-3b",
          "trainable": True,
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
    "epochs": 3,
    "batch_size": "auto",
  },
  "backend": {
    "type": "ray",
    #"cache_dir": "s3://ludwigconfig/ludwig_config/",
    # "loader": {
    #     "fully_executed": False,
    #     "window_size_bytes": 500000000,
    # },
    "trainer": {
      "strategy": "fsdp",
      "use_gpu": True,
      "num_workers": 2,
      "resources_per_worker": {
        "CPU": 4,
        "GPU": 1,
      }
    }
  }
}

model = LudwigModel(config,
                    logging_level=logging.INFO,
                    )
train_stats, preprocessed_data, output_directory = model.train(dataset='new.csv',skip_save_processed_input=True)
