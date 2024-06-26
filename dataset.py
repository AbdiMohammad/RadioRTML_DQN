import random
from collections import namedtuple
import numpy as np
import os
import json


class rf_env():

    FEConfig = namedtuple('FEConfig',
                        ('LNACurrent', 'MixerCurrent'))

    MixerConfig = namedtuple('MixerConfig',
                            ('Gain', 'NF', 'IIP3'))

    CommConfig = namedtuple('CommConfig', 
                            ('SignalPower', 'SNR'))

    SignalPower_List = [-75, -70, -65]
    SNR_List = [10, 20, 30]
    LNACurrent_List = [31.25, 100, 200, 250, 300, 400, 500] # [500, 31.25, 250]
    MixerCurrent_List = [50, 100, 200, 250, 300, 400] # [400, 50, 250]
    MixerConfig_Dict = {
        400: MixerConfig(18.5, 22.5, -3.88),
        300: MixerConfig(17.2, 24.3, -5.9),
        250: MixerConfig(16.52, 25.3, -5.9),
        200: MixerConfig(15.5, 26.5, -5.7),
        100: MixerConfig(12.2, 30, -3.5),
        50: MixerConfig(8.16, 34, -1.39)
    }

    PreambleLengthInBytes = 5
    SamplingFrequency = 54.4e6
    SymbolDuration = 2e-6
    BitsPerSymbol = 2
    PreambleLength = int(PreambleLengthInBytes * 8 / BitsPerSymbol * SymbolDuration * SamplingFrequency)

    @staticmethod
    def configStr(comm_config, fe_config):
        return f"{comm_config.SignalPower}_{comm_config.SNR}_{fe_config.LNACurrent}_{rf_env.MixerConfig_Dict[fe_config.MixerCurrent].Gain}_{rf_env.MixerConfig_Dict[fe_config.MixerCurrent].NF}_{rf_env.MixerConfig_Dict[fe_config.MixerCurrent].IIP3}"
    
    @staticmethod
    def load_all_measures(dataset_dir):
        measures_dict = {}
        for signal_power in rf_env.SignalPower_List:
            for snr in rf_env.SNR_List:
                for lna_current in rf_env.LNACurrent_List:
                    for mixer_current in rf_env.MixerCurrent_List:
                        comm_config = rf_env.CommConfig(signal_power, snr)
                        fe_config = rf_env.FEConfig(lna_current, mixer_current)
                        with open(os.path.join(dataset_dir, "MATLABGeneratedFiles", rf_env.configStr(comm_config, fe_config), "Measures.json")) as measures_file:
                            measures_dict[rf_env.configStr(comm_config, fe_config)] = json.load(measures_file)
                            measures_dict[rf_env.configStr(comm_config, fe_config)]['Power'] = rf_env.calcPower(lna_current, mixer_current)
        
        return measures_dict
    
    @staticmethod
    def calcPower(lna_current, mixer_current):
        return (lna_current * 1) + (mixer_current * 1.2)
    
    @staticmethod
    def normalize(array):
        if type(array) is not np.ndarray:
            array = np.array(array)
        scaled = (array - array.min()) / (array.max() - array.min())
        normalized = (scaled - scaled.mean()) / scaled.std() + scaled.mean()
        return normalized


    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        self.measures_dict = rf_env.load_all_measures(self.dataset_dir)

        rmsEVM_Normalized = rf_env.normalize([measures['rmsEVM'] for measures in self.measures_dict.values()])
        Power_Normalized = rf_env.normalize([measures['Power'] for measures in self.measures_dict.values()])
        for idx, config in enumerate(self.measures_dict.keys()):
            self.measures_dict[config]['rmsEVM_Normalized'] = rmsEVM_Normalized[idx]
            self.measures_dict[config]['Power_Normalized'] = Power_Normalized[idx]

        self.n_observations = rf_env.PreambleLength
        # FIXME: return signal power and snrs instead of preamble IQ symbols
        self.n_observations = 2

        self.comm_config = rf_env.CommConfig(None, None)

        self.n_actions = len(rf_env.LNACurrent_List) * len(rf_env.MixerCurrent_List)
        self.fe_config_idx = random.randint(0, self.n_actions - 1)

        self.step_idx = None
        self.MaxSteps = 100
    
    def improve_comm_env(self, action=None):
        terminated = False
        if self.step_idx is None:
            self.step_idx = 0
            self.comm_change_steps = list(np.linspace(0, self.MaxSteps, len(rf_env.SignalPower_List) * len(rf_env.SNR_List) + 1, dtype=np.int32))[1:-1] # random.sample(range(0, self.MaxSteps), len(rf_env.SignalPower_List) * len(rf_env.SNR_List))
            self.comm_change_steps.sort()
        else:
            self.step_idx += 1
            if self.step_idx > self.MaxSteps:
                terminated = True

        if action is not None:
            self.fe_config_idx = action

        if self.comm_config == rf_env.CommConfig(None, None):
            self.comm_config = rf_env.CommConfig(min(rf_env.SignalPower_List), min(rf_env.SNR_List))
        else:
            if len(self.comm_change_steps) > 0 and self.step_idx > self.comm_change_steps[0]:
                self.comm_change_steps.pop(0)

                comm_config_found = False
                for signal_power in rf_env.SignalPower_List:
                    for snr in rf_env.SNR_List:
                        if comm_config_found:
                            self.comm_config = rf_env.CommConfig(signal_power, snr)
                            comm_config_found = False
                            break
                        if rf_env.CommConfig(signal_power, snr) == self.comm_config:
                            comm_config_found = True
        
        observation = np.array([self.comm_config.SignalPower, self.comm_config.SNR])

        rmsEVM_Normalized = self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]['rmsEVM_Normalized']
        Power_Normalized = self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]['Power_Normalized']
        BER = self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]['BER']

        reward = 2 \
                 - rmsEVM_Normalized \
                 - Power_Normalized \
                 - (10e4 * BER)
        
        return observation, reward, terminated, self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]
    
    def reset(self, keep_last_config=False):
        self.step_idx = 0
        self.comm_change_steps = random.sample(range(0, self.MaxSteps), 3)
        self.comm_change_steps.sort()

        self.comm_config = rf_env.CommConfig(random.choice(rf_env.SignalPower_List), random.choice(rf_env.SNR_List))

        if not keep_last_config:
            self.fe_config_idx = random.randint(0, self.n_actions - 1)
        
        # FIXME: return signal power and snrs instead of preamble IQ symbols
        return np.array([self.comm_config.SignalPower, self.comm_config.SNR]), None
        return self.load_preamble(), None

    def step(self, action):
        if len(self.comm_change_steps) > 0 and self.step_idx > self.comm_change_steps[0]:
            self.comm_config = rf_env.CommConfig(random.choice(rf_env.SignalPower_List), random.choice(rf_env.SNR_List))
            self.comm_change_steps.pop(0)

        self.fe_config_idx = action
        # observation = self.load_preamble()
        # FIXME: return signal power and snrs instead of preamble IQ symbols
        observation = np.array([self.comm_config.SignalPower, self.comm_config.SNR])
        
        rmsEVM_Normalized = self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]['rmsEVM_Normalized']
        Power_Normalized = self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]['Power_Normalized']
        BER = self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]['BER']

        reward = - rmsEVM_Normalized \
                 - Power_Normalized \
                 - (10e4 * BER)

        terminated = False
        self.step_idx += 1
        if self.step_idx > self.MaxSteps:
            terminated = True
        
        return observation, reward, terminated, self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]

    def load_preamble(self, config):
        preabmle_I = np.loadtxt(os.path.join(self.dataset_dir, "CadenceGeneratedFiles_0", f"{self.signal_power}_{self.snr}_{config[0]}_{config[1].Gain}_{config[1].NF}_{config[1].IIP3}", "ISignal.csv"), skiprows=3, usecols=(1,), max_rows=self.n_observations)[np.newaxis, :]
        preabmle_Q = np.loadtxt(os.path.join(self.dataset_dir, "CadenceGeneratedFiles_0", f"{self.signal_power}_{self.snr}_{config[0]}_{config[1].Gain}_{config[1].NF}_{config[1].IIP3}", "QSignal.csv"), skiprows=3, usecols=(1,), max_rows=self.n_observations)[np.newaxis, :]
        return np.stack([preabmle_I, preabmle_Q])

    def getFEConfig(self):
        return rf_env.FEConfig(self.getLNACurrent(), self.getMixerCurrent())

    def getLNACurrent(self):
        return rf_env.LNACurrent_List[self.fe_config_idx // len(rf_env.MixerCurrent_List)]
    
    def getMixerCurrent(self):
        return rf_env.MixerCurrent_List[self.fe_config_idx % len(rf_env.MixerCurrent_List)]
    
    def getPower(self):
        return rf_env.calcPower(self.getLNACurrent(), self.getMixerCurrent())