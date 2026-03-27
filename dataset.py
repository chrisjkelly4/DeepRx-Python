import sionna
import config
import utils
import numpy as np
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, CDL, TDL
import sionna.phy.ofdm as phy
import h5py
import torch
import time

# Define Necessary Sionna Components
ofdm_resource_grid = sionna.phy.ofdm.ResourceGrid(num_ofdm_symbols=config.NUM_OFDM_SYMBOLS,
                                                  fft_size=config.FFT_SIZE,
                                                  subcarrier_spacing=config.SUBCARRIER_SPACING_HZ,
                                                  num_tx=1,
                                                  num_streams_per_tx=1,
                                                  cyclic_prefix_length=config.CYCLIC_PREFIX,
                                                  num_guard_carriers=(0, 0),
                                                  dc_null=False,
                                                  pilot_pattern='kronecker',
                                                  pilot_ofdm_symbol_indices=[2, 11],
                                                  precision=None,
                                                  device='cuda:0')

mapper = sionna.phy.mapping.Mapper(constellation_type='qam', num_bits_per_symbol=4)

# Panel array configuration for the transmitter and receiver
# Deep Rx uses SIMO so we set the Base Station (bs_array) to only have 1x1
# And the user terminal(ut_array) the user terminal to have a 1x2 dimensionality

bs_array = AntennaArray(num_rows=1,
                        num_cols=1,
                        polarization='dual',
                        polarization_type='cross',
                        antenna_pattern='38.901',
                        carrier_frequency=3.5e9)

ut_array = AntennaArray(num_rows=1,
                        num_cols=config.N_RX,
                        polarization='single',
                        polarization_type='V',
                        antenna_pattern='omni',
                        carrier_frequency=3.5e9)


def generate_channel_model(gen_type, rms_delay_spread, max_speed):
    if gen_type == 'val':
        options = ['CDL-A', 'CDL-E', 'TDL-A', 'TDL-E']
    else:
        options = ['CDL-B', 'CDL-C', 'CDL-D', 'TDL-B', 'TDL-C', 'TDL-D']

    model_index = np.random.randint(len(options))
    selected_model = options[model_index]

    if 'TDL' in selected_model:

        tdl = TDL(model=selected_model[-1],
                  delay_spread=rms_delay_spread,
                  carrier_frequency=config.CARRIER_FREQ_HZ,
                  min_speed=0.0,
                  max_speed=max_speed,
                  num_rx_ant=config.N_RX)
        channel_model = tdl

    else:
        cdl = CDL(model=selected_model[-1],
                  delay_spread=rms_delay_spread,
                  carrier_frequency=config.CARRIER_FREQ_HZ,
                  ut_array=ut_array,
                  bs_array=bs_array,
                  direction='uplink')
        channel_model = cdl

    return channel_model


def generate_batch():
    snr_db = np.random.uniform(config.SNR_DB_MIN, config.SNR_DB_MAX)  # dB
    sir_db = np.random.uniform(config.SIR_DB_MIN, config.SIR_DB_MAX)  # dB

    rms_delay_spread = np.random.uniform(config.RMS_DELAY_SPREAD_MIN, config.RMS_DELAY_SPREAD_MAX)  # seconds
    max_doppler = np.random.uniform(config.MAX_DOPPLER_MIN, config.MAX_DOPPLER_MAX)  # Hz

    # worth checking these two lines
    wavelength = config.SPEED_OF_LIGHT / config.CARRIER_FREQ_HZ  # 1.
    max_speed = max_doppler * wavelength  # 2.

    channel_model = generate_channel_model('val', rms_delay_spread, max_speed)

    channel = sionna.phy.channel.OFDMChannel(channel_model=channel_model,
                                             resource_grid=ofdm_resource_grid)

    ebno_db = snr_db - 10 * np.log10(config.MODULATION_ORDER * config.CODE_RATE)
    noise_variance = sionna.phy.utils.ebnodb2no(ebno_db,
                                                num_bits_per_symbol=config.MODULATION_ORDER,
                                                coderate=config.CODE_RATE,
                                                resource_grid=ofdm_resource_grid)
    # channel_model = RandomChoice(TRAIN_CHANNEL_MODELS)

    # dmrs_config = RandomChoice(DMRS_OPTIONS)  # 4 options per Fig 5
    # multiple DMRS configs may not be needed given minimal performance gain

    # 2. generate random bits
    num_bits = ofdm_resource_grid.num_data_symbols * config.MODULATION_ORDER
    bit_pattern = np.random.randint(0, 2, size=num_bits)

    # 2.1 Map to tensor
    bit_pattern_tensor = torch.tensor(bit_pattern)

    bit_pattern_tensor = torch.reshape(bit_pattern_tensor, [1, -1])  # add batch dimension

    # 3. Map tensor to QAM symbols

    symbols = mapper(bit_pattern_tensor)
    symbols = torch.reshape(symbols, [1, 1, 1, -1])  # [batch, tx, streams, symbols]

    # 4. create OFDM grid with DMRS pilots

    resource_grid_mapper = phy.ResourceGridMapper(ofdm_resource_grid)
    ofdm_grid = resource_grid_mapper(symbols)
    # 5. pass through Sionna channel
    Y = channel(ofdm_grid, noise_variance)
    # 5.1 Compute Xp from Y --------
    # Get pilot mask - shape (1, 1, 14, 312)
    mask = ofdm_resource_grid.pilot_pattern.mask

    # Get pilots - already in the right positions
    pilots = ofdm_resource_grid.pilot_pattern.pilots

    # Reshape mask to (14, 312)
    mask = torch.squeeze(mask)

    # Create Xp - zeros everywhere except pilot positions
    Xp = mask.to(torch.complex64)

    return Y, Xp, bit_pattern


class DeepRxDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self._hdf5_file = None
        with h5py.File(filepath, 'r') as f:  # open briefly just to get length
            self.length = len(f['bits'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._open_hdf5()  # opens once, then reuses forever
        Y = torch.tensor(self._hdf5_file['Y'][idx])
        Xp = torch.tensor(self._hdf5_file['Xp'][idx])
        bits = torch.tensor(self._hdf5_file['bits'][idx], dtype=torch.float32)

        # Z = utils.construct_input_tensor(Y, Xp) #removed for ablation study
        Z = utils.construct_input_tensor_ablation(Y, Xp)
        return Z, bits

    def _open_hdf5(self):
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.filepath, 'r')

    def __del__(self):
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset._open_hdf5()

if __name__ == '__main__':
    n_samples = 200000
    save_every = 10000

    with h5py.File('/workspace/Datasets/placholder_data.h5', 'w') as f:
        Y_ds = f.create_dataset('Y', shape=(n_samples, 1, 1, 2, 14, 512), dtype='complex64')
        Xp_ds = f.create_dataset('Xp', shape=(n_samples, 14, 512), dtype='complex64')
        bits_ds = f.create_dataset('bits', shape=(n_samples, 24576), dtype='int32')

        print("---- Starting Generation ----")
        start_time = time.time()
        for i in range(n_samples):
            Y, Xp, bits = generate_batch()
            Y_ds[i] = Y.cpu()
            Xp_ds[i] = Xp.cpu()
            bits_ds[i] = bits
            if i % save_every == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (n_samples - i) / rate
                print(f"Generated {i}/{n_samples} | {rate:.1f} samples/sec | ETA: {remaining / 3600:.1f} hrs")

        print("---- Generation Complete ----")