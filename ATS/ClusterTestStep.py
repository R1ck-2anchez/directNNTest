from ATS import ats_config
import torch


class ClusterTestStep(object):
    def split_data_region_with_idx(self, Tx_prob_matrixc, i, idx):
        #Tx_i_prob_vec = Tx_prob_matrixc[:, i]
        #print("idx before selection", idx)
        #print("Tx_prob_vec", Tx_i_prob_vec)
        S1_i =  torch.tensor([]) #Tx_prob_matrixc[Tx_i_prob_vec < ats_config.boundary]
        #print("tx < bd", Tx_prob_matrixc[Tx_i_prob_vec < ats_config.boundary])
        #print("S1_i", S1_i)
        idx_1 = torch.tensor([]) #idx[Tx_i_prob_vec < ats_config.boundary]
        #print("idx_1", idx_1)
        S0_i = Tx_prob_matrixc #[(Tx_i_prob_vec >= ats_config.boundary) & (Tx_i_prob_vec < ats_config.up_boundary)]
        #print("S0_i", S0_i)
        idx_0 = idx #[(Tx_i_prob_vec >= ats_config.boundary) & (Tx_i_prob_vec < ats_config.up_boundary)]
        #print("idx_0", idx_0)
        S2_i =  torch.tensor([]) #Tx_prob_matrixc[(Tx_i_prob_vec > ats_config.up_boundary)]
        idx_2 = torch.tensor([]) #idx[(Tx_i_prob_vec > ats_config.up_boundary)]

        return S2_i, idx_2, S0_i, idx_0, S1_i, idx_1

    def split_data_region(self, Tx_prob_matrixc, i):
        Tx_i_prob_vec = Tx_prob_matrixc[:, i]

        S1_i = Tx_prob_matrixc[Tx_i_prob_vec < ats_config.boundary]
        S0_i = Tx_prob_matrixc[(Tx_i_prob_vec >= ats_config.boundary) & (Tx_i_prob_vec < ats_config.up_boundary)]

        S2_i = Tx_prob_matrixc[(Tx_i_prob_vec > ats_config.up_boundary)]
        return S2_i, S0_i, S1_i