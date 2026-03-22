# CONSTANTS -----------

CARRIER_FREQ_HZ      = 4e9      # 4 GHz

TRAIN_CHANNEL_MODELS = ['CDL-B', 'CDL-C', 'CDL-D', 'TDL-B', 'TDL-C', 'TDL-D']
VAL_CHANNEL_MODELS   = ['CDL-A', 'CDL-E', 'TDL-A', 'TDL-E']

# Definie Min and Max fvalues for randomly sampled variables later on.
RMS_DELAY_SPREAD_MIN = 10e-9
RMS_DELAY_SPREAD_MAX = 300e-9
MAX_DOPPLER_MIN = 0
MAX_DOPPLER_MAX = 500

SPEED_OF_LIGHT = 3e8

SNR_DB_MIN, SNR_DB_MAX = -4, 32
SIR_DB_MIN, SIR_DB_MAX = 0, 36

N_PRB                = 26       # given in table
N_SUBCARRIERS_PER_PRB = 12     # this is fixed by 5G spec — always 12
SUBCARRIER_SPACING_HZ = 15000   # 15000Hz = 15 kHz — this is a 5G NR numerology mu=0 value

N_SUBCARRIERS = N_PRB * N_SUBCARRIERS_PER_PRB  # 26 * 12 = 312
SYMBOL_DURATION_S = 1 / SUBCARRIER_SPACING_HZ  # = 66.7 us (pure OFDM, no CP)
CYCLIC_PREFIX   = 4.69e-6 # µs    (standard 5G normal CP)

SYMBOL_DURATION = SYMBOL_DURATION_S + CYCLIC_PREFIX


TTI_DURATION_S = 1e-3          # 1ms, standard 5G slot

N_SYMBOLS_PER_TTI   = 14       # fixed by 5G NR slot structure
MODULATION_ORDER    = 4        # 16-QAM = 4 bits per symbol (log2(16))
CODE_RATE           = 658/1024  # directly from table
N_TX                = 1
N_RX                = 2

N_BITS_PER_SYMBOL   = 8
NUM_OFDM_SYMBOLS = 14
FFT_SIZE = 512