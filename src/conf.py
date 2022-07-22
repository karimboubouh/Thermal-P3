# DATA
DATA_DIR = "data/ecobee/"
DATA_CLUSTERS_DIR = "data/ecobee/clusters"
DATA_SIMILARITY_DIR = "data/ecobee/similarities"
META_CLUSTER_COLUMNS = ['Identifier', 'Floor Area [ft2]', 'Age of Home [years]']
META_SIMILARITY_COLUMNS = ['Identifier', 'Floor Area [ft2]', 'Number of Floors', 'Age of Home [years]']
DF_CLUSTER_COLUMNS = ['in_cool', 'in_heat', 'in_hum', 'out_hum', 'out_temp', 'in_temp']
TIME_ABSTRACTION = "5min"  # None | "H" | 15min
RECORD_PER_HOUR = 12  # 12 for 5min | 4 for 15min | 1 for 1H
WINTER = ['12', '01', '02']
SPRING = ['03', '04', '05']
SUMMER = ['06', '07', '08']
AUTUMN = ['09', '10', '11']

# NETWORK
PORT = 9000
LAUNCHER_PORT = 19491
TCP_SOCKET_BUFFER_SIZE = 5000000
TCP_SOCKET_SERVER_LISTEN = 10
SOCK_TIMEOUT = 20
LAUNCHER_TIMEOUT = 60

# ML
ML_ENGINE = "TensorFlow"  # "NumPy", "TensorFlow"
DEFAULT_VAL_DS = "val"
DEFAULT_MEASURE = "mean"
EVAL_ROUND = 10
TRAIN_VAL_TEST_RATIO = [.8, .1, .1]
RECORD_RATE = 10
M_CONSTANT = 1
WAIT_TIMEOUT = 10  # 1.5
WAIT_INTERVAL = 0.2  # 0.02
FUNC_TIMEOUT = 600
TEST_SCOPE = 'neighborhood'
IDLE_POWER = 12.60
INFERENCE_BATCH_SIZE = 256
DATASET_DUPLICATE = 0
