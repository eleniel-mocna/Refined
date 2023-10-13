# P2Rank models pipeline

This is a pipeline for comparing and using multiple models for P2Rank.

# Running the pipeline

Run configurations for this pipeline are governed by IntelliJ based IDEs. The most convenient way to run the pipeline is
to use PyCharm. But running via bash is also possible. All run scripts are also in the `run_configs` folder.

1. Download data
   from [MFF storage](https://cunicz-my.sharepoint.com/personal/89562630_cuni_cz/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F89562630%5Fcuni%5Fcz%2FDocuments%2FDavid%5Farffs&ga=1)(
   access permissions required) into `data/raw`
2. Extract this data into pickle files using `Extract arffs to pickle`.

# REFINED

## Usage of REFINED:

```(python)
refined = Refined(samples, rows, cols, "test_folder")
refined.run()
refined.transform(samples)
```

Already trained REFINED can be used by loading `best_individual.npy`
from the specified folder. Then to use this to transform samples, use

```(python)
Refined#transform_from_vector(samples: np.ndarray,
                              refined_vector: np.ndarray,
                              rows: int,
                              cols: int)
```