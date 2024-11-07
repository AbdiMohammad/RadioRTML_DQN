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
    MaxSteps = 100

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
    def load_all_preambles(dataset_dir):
        preambles_dict = {}
        for signal_power in rf_env.SignalPower_List:
            for snr in rf_env.SNR_List:
                for lna_current in rf_env.LNACurrent_List:
                    for mixer_current in rf_env.MixerCurrent_List:
                        comm_config = rf_env.CommConfig(signal_power, snr)
                        fe_config = rf_env.FEConfig(lna_current, mixer_current)
                        preabmle_I = np.loadtxt(os.path.join(dataset_dir, "CadenceGeneratedFiles_0", rf_env.configStr(comm_config, fe_config), "ISignal.csv"), skiprows=3, usecols=(1,), max_rows=rf_env.PreambleLength)[np.newaxis, :]
                        preabmle_Q = np.loadtxt(os.path.join(dataset_dir, "CadenceGeneratedFiles_0", rf_env.configStr(comm_config, fe_config), "QSignal.csv"), skiprows=3, usecols=(1,), max_rows=rf_env.PreambleLength)[np.newaxis, :]
                        preambles_dict[rf_env.configStr(comm_config, fe_config)] = np.stack([preabmle_I, preabmle_Q])
        
        return preambles_dict
    
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
        # self.preambles_dict = rf_env.load_all_preambles(self.dataset_dir)

        rmsEVM_Normalized = rf_env.normalize([measures['rmsEVM'] for measures in self.measures_dict.values()])
        Power_Normalized = rf_env.normalize([measures['Power'] for measures in self.measures_dict.values()])
        BER_Normalized = rf_env.normalize([measures['BER'] for measures in self.measures_dict.values()])
        for idx, config in enumerate(self.measures_dict.keys()):
            self.measures_dict[config]['rmsEVM_Normalized'] = rmsEVM_Normalized[idx]
            self.measures_dict[config]['Power_Normalized'] = Power_Normalized[idx]
            self.measures_dict[config]['BER_Normalized'] = BER_Normalized[idx]

        self.n_observations = rf_env.PreambleLength
        # FIXME: return signal power and snrs instead of preamble IQ symbols
        self.n_observations = 2

        self.comm_config = rf_env.CommConfig(None, None)

        self.n_actions = len(rf_env.LNACurrent_List) * len(rf_env.MixerCurrent_List)
        self.fe_config_idx = random.randint(0, self.n_actions - 1)

        self.step_idx = None
        rf_env.MaxSteps = 100
    
    def improve_comm_env(self, action=None):
        terminated = False
        if self.step_idx is None:
            self.step_idx = 0
            self.comm_change_steps = list(np.linspace(0, rf_env.MaxSteps, len(rf_env.SignalPower_List) * len(rf_env.SNR_List) + 1, dtype=np.int32))[1:-1] # random.sample(range(0, rf_env.MaxSteps), len(rf_env.SignalPower_List) * len(rf_env.SNR_List))
            self.comm_change_steps.sort()
        else:
            self.step_idx += 1
            if self.step_idx > rf_env.MaxSteps:
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
        
        # FIXME: return signal power and snrs instead of preamble IQ symbols
        observation = np.array([self.comm_config.SignalPower, self.comm_config.SNR])
        # observation = self.get_preamble()

        rmsEVM_Normalized = self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]['rmsEVM_Normalized']
        Power_Normalized = self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]['Power_Normalized']
        BER_Normalized = self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]['BER_Normalized']

        reward = - (1 * rmsEVM_Normalized) \
                 - (3 * Power_Normalized) \
                 - (10 * BER_Normalized)
        
        return observation, reward, terminated, self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]
    
    def reset(self, keep_last_config=False):
        self.step_idx = 0
        self.comm_change_steps = random.sample(range(0, rf_env.MaxSteps), 3)
        self.comm_change_steps.sort()

        self.comm_config = rf_env.CommConfig(random.choice(rf_env.SignalPower_List), random.choice(rf_env.SNR_List))

        if not keep_last_config:
            self.fe_config_idx = random.randint(0, self.n_actions - 1)
        
        # FIXME: return signal power and snrs instead of preamble IQ symbols
        return np.array([self.comm_config.SignalPower, self.comm_config.SNR]), None
        # return self.get_preamble(), None

    def step(self, action):
        if len(self.comm_change_steps) > 0 and self.step_idx > self.comm_change_steps[0]:
            self.comm_config = rf_env.CommConfig(random.choice(rf_env.SignalPower_List), random.choice(rf_env.SNR_List))
            self.comm_change_steps.pop(0)

        self.fe_config_idx = action
        # observation = self.get_preamble()
        # FIXME: return signal power and snrs instead of preamble IQ symbols
        observation = np.array([self.comm_config.SignalPower, self.comm_config.SNR])
        
        rmsEVM_Normalized = self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]['rmsEVM_Normalized']
        Power_Normalized = self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]['Power_Normalized']
        BER_Normalized = self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]['BER_Normalized']

        reward = - (1 * rmsEVM_Normalized) \
                 - (3 * Power_Normalized) \
                 - (10 * BER_Normalized)

        terminated = False
        self.step_idx += 1
        if self.step_idx > rf_env.MaxSteps:
            terminated = True
        
        return observation, reward, terminated, self.measures_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]

    def get_preamble(self):
        return self.preambles_dict[rf_env.configStr(self.comm_config, self.getFEConfig())]

    def getFEConfig(self):
        return rf_env.FEConfig(self.getLNACurrent(), self.getMixerCurrent())

    def getLNACurrent(self):
        return rf_env.LNACurrent_List[self.fe_config_idx // len(rf_env.MixerCurrent_List)]
    
    def getMixerCurrent(self):
        return rf_env.MixerCurrent_List[self.fe_config_idx % len(rf_env.MixerCurrent_List)]
    
    def getPower(self):
        return rf_env.calcPower(self.getLNACurrent(), self.getMixerCurrent())