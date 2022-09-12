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