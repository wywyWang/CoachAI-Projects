"""Rally classifier models."""
from typing import Tuple, Dict, Any
import tensorflow as tf
import keras_self_attention
from prosenet.model import ProSeNet
from keras_ordered_neurons import ONLSTM
from keras_pos_embd import TrigPosEmbedding
from keras_transformer import get_encoders
from keras_transformer.gelu import gelu
import custom_layers
import cnn


def bad_net(shot_sequence_shape: Tuple[int, int],
            embed_types_size: int = None,
            embed_area_size: int = None,
            rally_info_shape: int = None,
            cnn_kwargs: Dict[str, Any] = {'filters': 32, 'kernel_size': 3},
            rnn_kwargs: Dict[str, Any] = {'units': 32},
            attention_kwargs: Dict[str, Any] = {},
            dense_kwargs: Dict[str, Any] = {}
            ) -> tf.keras.Model:
    """Create BadNet(our proposed model) for rally classification."""
    # Layers
    input_hit_area = tf.keras.Input(shape=shot_sequence_shape[0], name='Hit_area_input')
    input_player_area = tf.keras.Input(shape=shot_sequence_shape[0], name='Player_area_input')
    input_opponent_area = tf.keras.Input(shape=shot_sequence_shape[0], name='Opponent_area_input')
    
    if embed_area_size is not None:
        area_embedding = tf.keras.layers.Embedding(input_dim=embed_area_size, output_dim=10, mask_zero=True, name='Area_embedding')
    else:
        area_embedding = None

    input_shots = tf.keras.Input(shape=shot_sequence_shape, name='Shots_input')
    shots_concat_areas = tf.keras.layers.Concatenate(name='Shots_areas_merging')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')

    if embed_types_size:
        input_shot_types = tf.keras.Input(shape=shot_sequence_shape[0], name='Shot_types_input')
        layer_embedding = tf.keras.layers.Embedding(input_dim=embed_types_size, output_dim=15, mask_zero=True, name='Shot_types_embedding')
        shot_mu_embedding = tf.keras.layers.Embedding(input_dim=embed_types_size, output_dim=15, mask_zero=True, name='Time_influence_occurrence')
        shot_theta_embedding = tf.keras.layers.Embedding(input_dim=embed_types_size, output_dim=15, mask_zero=True, name='Time_influence_shot')
        
        input_time_proportion = tf.keras.Input(shape=shot_sequence_shape[0], name='Time_proportion_input')
        time_multiplication = tf.keras.layers.Multiply(name='Time_proportion_multiply')
        time_addition = tf.keras.layers.Add(name='Time_proportion_add')
        time_activation = tf.keras.layers.Activation('sigmoid', name='Time_activation')

        activity_embedding = tf.keras.layers.Multiply(name='Shots_time_multiply')
        layer_concat_embedding = tf.keras.layers.Concatenate(name='Shots_features_merging')
    else:
        input_shot_types = None
        layer_embedding = None
        layer_concat_embedding = None
        shot_mu_embedding = None
        shot_theta_embedding = None
        input_time_proportion = None
        time_multiplication = None
        time_addition = None
        time_activation = None
        activity_embedding = None
        layer_concat_embedding = None

    layer_cnn = custom_layers.StaggeredConv1D(name='Local_pattern_extration', **cnn_kwargs)
    layer_rnn = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(return_sequences=True, **rnn_kwargs), name='Bidirectional_recurrent_layer')
    layer_concat_cnn_rnn = tf.keras.layers.Concatenate(name='Patterns_states_merging')
    layer_attention = keras_self_attention.SeqWeightedAttention(return_attention=True, **attention_kwargs)

    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=rally_info_shape, name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None
    layer_dense = tf.keras.layers.Dense(units=1, activation='sigmoid', **dense_kwargs)

    # Forward pass
    inputs = [input_hit_area, input_player_area, input_opponent_area]
    embeded_hit_area = area_embedding(input_hit_area)
    embeded_player_area = area_embedding(input_player_area)
    embeded_opponent_area = area_embedding(input_opponent_area)

    inputs.append(input_shots)
    input_shots_concat_area = shots_concat_areas([embeded_hit_area, embeded_player_area, embeded_opponent_area, input_shots])
    masked_sequence = layer_masking(input_shots_concat_area)

    if embed_types_size is not None:
        inputs.append(input_shot_types)
        embeded_shot_types = layer_embedding(input_shot_types)
        embeded_shot_mu = shot_mu_embedding(input_shot_types)
        embeded_shot_theta = shot_theta_embedding(input_shot_types)

        inputs.append(input_time_proportion)
        time_mu_proportion = time_multiplication([embeded_shot_mu, tf.tile(tf.expand_dims(input_time_proportion, -1), [1, 1, 15])])
        temporal_score = time_addition([embeded_shot_theta, time_mu_proportion])
        temporal_score = time_activation(temporal_score)

        embeded_activity = activity_embedding([temporal_score, embeded_shot_types])
        masked_sequence = layer_concat_embedding([masked_sequence, embeded_activity])

    pattern_sequence = layer_cnn(masked_sequence)
    hidden_states = layer_rnn(pattern_sequence)
    patterns_states = layer_concat_cnn_rnn([pattern_sequence, hidden_states])

    rally_represent, contributions = layer_attention(patterns_states)
    if rally_info_shape is not None:
        inputs.append(input_rally)
        rally_represent = layer_concat_rally([rally_represent, input_rally])
    output_win_prob = layer_dense(rally_represent)

    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    model_attention = tf.keras.Model(inputs=inputs, outputs=contributions)
    return model_predict, model_attention


def rnn(shot_sequence_shape: Tuple[int, int],
        rally_info_shape: int = None,
        rnn_structure: str = 'lstm',
        bidirectional_rnn: bool = True,
        rnn_kwargs: Dict[str, Any] = {'units': 32},
        dense_kwargs: Dict[str, Any] = {}
        ) -> tf.keras.Model:
    """Create a RNN-based rally classifier model."""
    rnns = {'gru': tf.keras.layers.GRU,
            'lstm': tf.keras.layers.LSTM}
    # Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape,
                                 name='Shots_input')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')
    rnn = rnns.get(rnn_structure, list(rnns.values())[0])
    layer_rnn = rnn(name='Recurrent_layer', **rnn_kwargs)
    if bidirectional_rnn:
        layer_rnn = tf.keras.layers.Bidirectional(
            layer_rnn, name='Bidirectional_recurrent_layer')
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=rally_info_shape,
                                     name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(
            name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None
    layer_dense = tf.keras.layers.Dense(units=1, activation='sigmoid',
                                        **dense_kwargs)
    # Forward pass
    inputs = [input_shots]
    masked_sequence = layer_masking(input_shots)
    rally_represent = layer_rnn(masked_sequence)
    if rally_info_shape is not None:
        inputs.append(input_rally)
        rally_represent = layer_concat_rally([rally_represent, input_rally])
    output_win_prob = layer_dense(rally_represent)
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    return model_predict


def deepmoji(shot_sequence_shape: Tuple[int, int],
             rally_info_shape: int = None,
             rnn_kwargs: Dict[str, Any] = {'units': 32},
             attention_kwargs: Dict[str, Any] = {},
             dense_kwargs: Dict[str, Any] = {}
             ) -> tf.keras.Model:
    """Create DeepMoji rally classifier model."""
    # Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape,
                                 name='Shots_input')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')
    layer_rnn1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(return_sequences=True, **rnn_kwargs),
        name='Bidirectional_recurrent_layer1')
    layer_rnn2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(return_sequences=True, **rnn_kwargs),
        name='Bidirectional_recurrent_layer2')
    layer_concat_input_rnn = tf.keras.layers.Concatenate(
        name='Input_rnn_merging')
    layer_attention = keras_self_attention.SeqWeightedAttention(
        return_attention=True, **attention_kwargs)
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=rally_info_shape,
                                     name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(
            name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None
    layer_dense = tf.keras.layers.Dense(units=1, activation='sigmoid',
                                        **dense_kwargs)
    # Forward pass
    inputs = [input_shots]
    masked_sequence = layer_masking(input_shots)
    hidden_states = layer_rnn2(layer_rnn1(masked_sequence))
    input_states = layer_concat_input_rnn([masked_sequence,
                                           hidden_states])
    rally_represent, contributions = layer_attention(input_states)
    if rally_info_shape is not None:
        inputs.append(input_rally)
        rally_represent = layer_concat_rally([rally_represent, input_rally])
    output_win_prob = layer_dense(rally_represent)
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    model_attention = tf.keras.Model(inputs=inputs, outputs=contributions)
    return model_predict, model_attention


def prosenet(shot_sequence_shape: Tuple[int, int],
             prosenet_kwargs: Dict[str, Any] = {'k': 100},
             rnn_kwargs: Dict[str, Any] = {'layer_type' : 'lstm',
                                           'layer_args' : {},
                                           'layers' : [32, 32],
                                           'bidirectional' : True}
             ) -> tf.keras.Model:
    """Create a ProSeNet rally classifier model."""
    model_predict = ProSeNet(input_shape=shot_sequence_shape, nclasses=2,
                             rnn_args=rnn_kwargs, **prosenet_kwargs)
    return model_predict


def onlstm(shot_sequence_shape: Tuple[int, int],
           rally_info_shape: int = None,
           onlstm_kwargs: Dict[str, Any] = {'units': 32, 'chunk_size': 4},
           dense_kwargs: Dict[str, Any] = {}
           ) -> tf.keras.Model:
    """Create an ON-LSTM rally classifier model."""
    rnns = {'gru': tf.keras.layers.GRU,
            'lstm': tf.keras.layers.LSTM}
    # Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape,
                                 name='Shots_input')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')
    layer_onlstm = ONLSTM(name='ONLSTM', **onlstm_kwargs)
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=rally_info_shape,
                                     name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(
            name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None
    layer_dense = tf.keras.layers.Dense(units=1, activation='sigmoid',
                                        **dense_kwargs)
    # Forward pass
    inputs = [input_shots]
    masked_sequence = layer_masking(input_shots)
    rally_represent = layer_onlstm(masked_sequence)
    if rally_info_shape is not None:
        inputs.append(input_rally)
        rally_represent = layer_concat_rally([rally_represent, input_rally])
    output_win_prob = layer_dense(rally_represent)
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    return model_predict


def transformer(shot_sequence_shape: Tuple[int, int],
                rally_info_shape: int = None,
                transformer_kwargs: Dict[str, Any] = {'encoder_num': 2,
                                                      'head_num': 2,
                                                      'hidden_dim': 32,
                                                      'feed_forward_activation': gelu},
                dense_kwargs: Dict[str, Any] = {}
           ) -> tf.keras.Model:
    """Create an ON-LSTM rally classifier model."""
    rnns = {'gru': tf.keras.layers.GRU,
            'lstm': tf.keras.layers.LSTM}
    # Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape,
                                 name='Shots_input')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')
    layer_pos_embed = TrigPosEmbedding(mode=TrigPosEmbedding.MODE_ADD)
    layer_pooling = tf.keras.layers.GlobalMaxPooling1D()
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=rally_info_shape,
                                     name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(
            name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None
    layer_dense = tf.keras.layers.Dense(units=1, activation='sigmoid',
                                        **dense_kwargs)
    # Forward pass
    inputs = [input_shots]
    masked_sequence = layer_masking(input_shots)
    pos_embed_seq = layer_pos_embed(masked_sequence)
    encoder_result = get_encoders(input_layer=pos_embed_seq, **transformer_kwargs)
    rally_represent = layer_pooling(encoder_result)
    if rally_info_shape is not None:
        inputs.append(input_rally)
        rally_represent = layer_concat_rally([rally_represent, input_rally])
    output_win_prob = layer_dense(rally_represent)
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    return model_predict
