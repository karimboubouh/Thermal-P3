import src.conf as C

if C.ML_ENGINE.lower() == "tensorflow":
    print("Already loading tensorflow")
    from src.ml.tensorflow.helpers import *
    from src.ml.numpy.datasets import get_dataset, train_val_test, inference_ds
elif C.ML_ENGINE.lower() == "numpy":
    print("Already loading numpy")
    from src.ml.numpy.models import *
    from src.ml.numpy.helpers import *
    from src.ml.numpy.datasets import get_dataset, train_val_test, inference_ds
else:
    exit(f'Unknown "{C.ML_ENGINE}" ML engine !')
