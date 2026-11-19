"""Parameter key names shared by data products and capture actions."""

SAMPLE_RATE_KEY = "sample_rate"
DURATION_MS_KEY = "duration_ms"
FFT_SIZE_KEY = "fft_size"
APD_BIN_SIZE_DB_KEY = "apd_bin_size_dB"
APD_MIN_BIN_DBM_KEY = "apd_min_bin_dBm"
APD_MAX_BIN_DBM_KEY = "apd_max_bin_dBm"
TD_BIN_SIZE_MS_KEY = "td_bin_size_ms"
PFP_FRAME_PERIOD_MS_KEY = "pfp_frame_period_ms"
DATA_PRODUCTS_KEY = "data_products"
RECORDING_ID_KEY = "recording_id"

# Parameter names that may be overridden for an individual capture.
UPDATABLE_KEYS = (
    SAMPLE_RATE_KEY,
    DURATION_MS_KEY,
    FFT_SIZE_KEY,
    APD_BIN_SIZE_DB_KEY,
    APD_MIN_BIN_DBM_KEY,
    APD_MAX_BIN_DBM_KEY,
    TD_BIN_SIZE_MS_KEY,
    PFP_FRAME_PERIOD_MS_KEY,
)
