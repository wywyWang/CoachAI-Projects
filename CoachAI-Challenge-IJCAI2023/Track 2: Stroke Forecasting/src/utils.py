import os
import torch
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


def draw_loss(record_total_loss, config):
    x_steps = range(1, config['epochs']+1, 20)
    fig = plt.figure(figsize=(12, 6))
    plt.title("{} loss".format(config['model_type']))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim(0, 6)
    plt.xticks(x_steps)
    plt.grid()
    plt.plot(record_total_loss['total'], label='Train total loss')
    plt.plot(record_total_loss['shot'], label='Train shot CE loss')
    plt.plot(record_total_loss['area'], label='Train area NLL loss')

    plt.legend()
    plt.savefig(f"{config['output_folder_name']}/loss.png")
    plt.close(fig)


def save(encoder, decoder, config, epoch=None):
    output_folder_name = f"{config['output_folder_name']}/"
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    if epoch is None:
        encoder_name = output_folder_name + 'encoder'
        decoder_name = output_folder_name + 'decoder'
        config_name = output_folder_name + 'config'
    else:
        encoder_name = output_folder_name + str(epoch) + 'encoder'
        decoder_name = output_folder_name + str(epoch) + 'decoder'
        config_name = output_folder_name + str(epoch) + 'config'
    
    torch.save(encoder.state_dict(), encoder_name)
    torch.save(decoder.state_dict(), decoder_name)
    with open(config_name, 'w') as config_file:
        config_file.write(str(config))