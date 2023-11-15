from torch.utils.data import Dataset
import numpy as np


PAD = 0


class BadmintonDataset(Dataset):
    def __init__(self, matches, encode_length, max_ball_round=20):
        super().__init__()
        self.max_ball_round = max_ball_round

        # convert data to (ball_round, shot_type, landing_x, landing_y, player, sets), rally_id
        self.sequences, self.rally_ids = {}, []
        for i, rally_id in enumerate(matches.index):
            ball_round, shot_type, landing_x, landing_y, player, sets = matches[rally_id]

            # filter less than encoding length shot in a rally
            if len(shot_type) <= encode_length:
                # record data only if current round(rally_id) last for more than encode_length times
                continue
            else:
                # standardize + relative = worse
                # landing_x[1:] = landing_x[1:] - landing_x[:-1]
                # landing_x[0] = 0
                # landing_y[1:] = landing_y[1:] - landing_y[:-1]
                # landing_y[0] = 0
                self.sequences[rally_id] = (
                    ball_round, shot_type, landing_x, landing_y, player, sets)

            self.rally_ids.append(rally_id)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        # get the data of 'index' round of data, return 'index' round as input and 'index+1' as output
        rally_id = self.rally_ids[index]
        ball_round, shot_type, landing_x, landing_y, player, sets = self.sequences[rally_id]

        # init data with fix length self.max_ball_round, with value = PAD(i.e. 0)
        pad_input_shot    = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_input_x       = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_input_y       = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_input_player  = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_shot   = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_x      = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_output_y      = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_output_player = np.full(self.max_ball_round, fill_value=PAD, dtype=int)

        # if ask for the ball_round that is out of range, return the full array
        # the following array input will have data index eq to output - 1,
        # which mean use 'index' ball should get predict output 'index+1'
        if len(ball_round) > self.max_ball_round:
            rally_len = self.max_ball_round

            # 0, 1, ..., max_ball_round-1
            pad_input_shot[:]   = shot_type[0:-1:1][:rally_len]
            pad_input_x[:]      = landing_x[0:-1:1][:rally_len]
            pad_input_y[:]      = landing_y[0:-1:1][:rally_len]
            pad_input_player[:] =    player[0:-1:1][:rally_len]
            # 1, 2, ..., max_ball_round
            pad_output_shot[:]   = shot_type[1::1][:rally_len]
            pad_output_x[:]      = landing_x[1::1][:rally_len]
            pad_output_y[:]      = landing_y[1::1][:rally_len]
            pad_output_player[:] =    player[1::1][:rally_len]
        else:
            # 0 ~ (n-2)
            rally_len = len(ball_round) - 1

            # 0, 1, ..., n-1
            pad_input_shot[:rally_len] = shot_type[0:-1:1]
            pad_input_x[:rally_len] = landing_x[0:-1:1]
            pad_input_y[:rally_len] = landing_y[0:-1:1]
            pad_input_player[:rally_len] = player[0:-1:1]
            # 1, 2, ..., n
            pad_output_shot[:rally_len] = shot_type[1::1]
            pad_output_x[:rally_len] = landing_x[1::1]
            pad_output_y[:rally_len] = landing_y[1::1]
            pad_output_player[:rally_len] = player[1::1]

        return (pad_input_shot, pad_input_x, pad_input_y, pad_input_player,
                pad_output_shot, pad_output_x, pad_output_y, pad_output_player,
                rally_len, sets[0])
        # set is current set, since holl round will belong to same set, simply return first term
