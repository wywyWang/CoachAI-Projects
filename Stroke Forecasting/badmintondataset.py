from torch.utils.data import Dataset
import numpy as np


PAD = 0


class BadmintonDataset(Dataset):
    def __init__(self, matches, encode_length, max_ball_round=20):
        super().__init__()
        self.max_ball_round = max_ball_round

        self.sequences, self.rally_ids = {}, []
        for i, rally_id in enumerate(matches.index):
            ball_round, shot_type, landing_x, landing_y, player, sets = matches[rally_id]
            
            # filter less than encoding length shot in a rally
            if len(shot_type) <= encode_length:
                continue
            else:
                # standardize + relative = worse
                # landing_x[1:] = landing_x[1:] - landing_x[:-1]
                # landing_x[0] = 0
                # landing_y[1:] = landing_y[1:] - landing_y[:-1]
                # landing_y[0] = 0
                self.sequences[rally_id] = (ball_round, shot_type, landing_x, landing_y, player, sets)

            self.rally_ids.append(rally_id)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        rally_id = self.rally_ids[index]
        ball_round, shot_type, landing_x, landing_y, player, sets = self.sequences[rally_id]

        pad_input_shot = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_input_x = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_input_y = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_input_player = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_shot = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_x = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_output_y = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_output_player = np.full(self.max_ball_round, fill_value=PAD, dtype=int)

        if len(ball_round) > self.max_ball_round:
            rally_len = self.max_ball_round

            pad_input_shot[:] = shot_type[0:-1:1][:rally_len]                                   # 0, 1, ..., max_ball_round-1
            pad_input_x[:] = landing_x[0:-1:1][:rally_len]
            pad_input_y[:] = landing_y[0:-1:1][:rally_len]
            pad_input_player[:] = player[0:-1:1][:rally_len]
            pad_output_shot[:] = shot_type[1::1][:rally_len]                                    # 1, 2, ..., max_ball_round
            pad_output_x[:] = landing_x[1::1][:rally_len]
            pad_output_y[:] = landing_y[1::1][:rally_len]
            pad_output_player[:] = player[1::1][:rally_len]
        else:
            rally_len = len(ball_round) - 1                                                     # 0 ~ (n-2)
            
            pad_input_shot[:rally_len] = shot_type[0:-1:1]                                      # 0, 1, ..., n-1
            pad_input_x[:rally_len] = landing_x[0:-1:1]
            pad_input_y[:rally_len] = landing_y[0:-1:1]
            pad_input_player[:rally_len] = player[0:-1:1]
            pad_output_shot[:rally_len] = shot_type[1::1]                                       # 1, 2, ..., n
            pad_output_x[:rally_len] = landing_x[1::1]
            pad_output_y[:rally_len] = landing_y[1::1]
            pad_output_player[:rally_len] = player[1::1]

        return (pad_input_shot, pad_input_x, pad_input_y, pad_input_player,
                pad_output_shot, pad_output_x, pad_output_y, pad_output_player,
                rally_len, sets[0])