from torch.utils.data import Dataset
import numpy as np
import pandas as pd

PAD = 0


class BadmintonDataset(Dataset):
    def __init__(self, data, used_column, args):
        super(BadmintonDataset).__init__()
        self.player_sequence = []
        self.court_sequence = []
        self.target_sequence = []

        self.encode_length = args['max_length']
        rally_id_grouped = data.groupby('rally_id').groups

        rally_data = data[used_column]        

        for rally_id in rally_id_grouped.values():
            rally_id = rally_id.to_numpy()
            one_rally = rally_data.iloc[rally_id].reset_index(drop=True)
            
            if len(one_rally) < args['encode_length'] + 1:
                continue
            
            if len(one_rally) > self.encode_length:
                seqence_length = self.encode_length
                one_rally = one_rally.head(self.encode_length + 1)
            else:
                seqence_length = len(one_rally) - 1
                
            # rally information
            player = one_rally[['player']].values[:-1].reshape(-1)
            shot_type = one_rally[['type']].values[:-1].reshape(-1)

            player_x = one_rally[['player_location_x']].values[:-1].reshape(-1)
            player_y = one_rally[['player_location_y']].values[:-1].reshape(-1)
            opponent_x = one_rally[['opponent_location_x']].values[:-1].reshape(-1)
            opponent_y = one_rally[['opponent_location_y']].values[:-1].reshape(-1)

            player = np.pad(player, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            shot_type  = np.pad(shot_type , (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            player_x = np.pad(player_x, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            player_y = np.pad(player_y, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            opponent_x = np.pad(opponent_x, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            opponent_y = np.pad(opponent_y, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))           
            
            player_A_x = np.empty((args['max_length'],), dtype=float)
            player_A_x[0::2] = player_x[0::2]
            player_A_x[1::2] = opponent_x[1::2]
            player_A_y = np.empty((args['max_length'],), dtype=float)
            player_A_y[0::2] = player_y[0::2]
            player_A_y[1::2] = opponent_y[1::2]

            player_B_x = np.empty((args['max_length'],), dtype=float)
            player_B_x[0::2] = opponent_x[0::2]
            player_B_x[1::2] = player_x[1::2]
            player_B_y = np.empty((args['max_length'],), dtype=float)
            player_B_y[0::2] = opponent_y[0::2]
            player_B_y[1::2] = player_y[1::2]

            self.player_sequence.append([player, shot_type, player_A_x, player_A_y, player_B_x, player_B_y, seqence_length])
            
            # predict target
            target_player_x = one_rally[['player_location_x']].values[1:].reshape(-1)
            target_player_y = one_rally[['player_location_y']].values[1:].reshape(-1)
            target_opponent_x = one_rally[['opponent_location_x']].values[1:].reshape(-1)
            target_opponent_y = one_rally[['opponent_location_y']].values[1:].reshape(-1)   
            target_type = one_rally[['type']].values[1:].reshape(-1)

            target_type = np.pad(target_type , (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            target_player_x = np.pad(target_player_x, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            target_player_y = np.pad(target_player_y, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            target_opponent_x = np.pad(target_opponent_x, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            target_opponent_y = np.pad(target_opponent_y, (0, self.encode_length - seqence_length), 'constant', constant_values=(0)) 

            target_A_x = np.empty((args['max_length'],), dtype=float)
            target_A_x[0::2] = target_opponent_x[0::2]
            target_A_x[1::2] = target_player_x[1::2]
            target_A_y = np.empty((args['max_length'],), dtype=float)
            target_A_y[0::2] = target_opponent_y[0::2]
            target_A_y[1::2] = target_player_y[1::2]

            target_B_x = np.empty((args['max_length'],), dtype=float)
            target_B_x[0::2] = target_player_x[0::2]
            target_B_x[1::2] = target_opponent_x[1::2]
            target_B_y = np.empty((args['max_length'],), dtype=float)
            target_B_y[0::2] = target_player_y[0::2]
            target_B_y[1::2] = target_opponent_y[1::2]

            self.target_sequence.append([target_A_x, target_A_y, target_B_x, target_B_y, target_type])
 
    def __len__(self):
        return len(self.player_sequence)
    
    def __getitem__(self, index):
        rally = self.player_sequence[index]
        target = self.target_sequence[index]
        
        return rally, target