import os
import sys
module_path = os.path.abspath(os.path.join('/home/grail/willaria_research/hobbes/calvin_models'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn
import torch
from typing import Dict, Tuple
from calvin_agent.models.perceptual_encoders.vision_network import VisionNetwork


# https://fairseq.readthedocs.io/en/latest/tutorial_simple_lstm.html


class HobbesLSTM2(pl.LightningModule):
    def __init__(
        self, static_img_embed_dim=64, gripper_img_embed_dim=32, encoder_hidden_dim=64, decoder_hidden_dim=64, dropout=0.1,
    ):
        super().__init__()

        # encoder for images (batch, 3, 200, 200) -> (batch, embed_dim(64))
        self.static_img_encoder = VisionNetwork(200, 200, 'LeakyReLU', 0.0, True, static_img_embed_dim, 3)

        self.gripper_img_encoder = VisionNetwork(84, 84, 'LeakyReLU', 0.0, True, gripper_img_embed_dim, 3)

        self.dropout = nn.Dropout(p=dropout)

        # LSTM https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # (batch, img_embed_dim(96) + robot_obs(15)) -> (batch, encoder_hidden_dim(64))
        self.lstm_encoder = nn.LSTM(input_size=static_img_embed_dim + gripper_img_embed_dim + 15,
                                    hidden_size=encoder_hidden_dim,
                                    num_layers=1,
                                    batch_first=True)

        # (batch, img_embed_dim(96) + encoder_hidden_dim(64)) -> (batch, decoder_hidden_dim(64))
        self.lstm_decoder = nn.LSTM(input_size=static_img_embed_dim + gripper_img_embed_dim + encoder_hidden_dim + 15,
                                    hidden_size=decoder_hidden_dim,
                                    num_layers=1,
                                    batch_first=True)

        # Define the output projection to our action space (batch, hidden_dim(64)) -> (batch, action_space(7))
        self.output_projection = nn.Linear(decoder_hidden_dim, 7)

        # Normalize to range (-1, 1) for actions
        self.normalize = nn.Tanh()

    # Assuming we are given a sequence of demonstration images, and one single image from runtime
    def forward(self, demonstration_observations, demonstration_observation_num, runtime_observation):
        ######################################## ENCODER ########################################
        # Encode demonstration input images (batch, img_embed_dim(64))
        # Confound sequence and batch for Conv2D https://discuss.pytorch.org/t/processing-sequence-of-images-with-conv2d/37494

        """ TODO: make this work probably.... all the data is laid out already

            observation: {
                'rgb_static': [Tensor(3,200,200), ...],
                'rgb_gripper': [Tensor(3,84,84), ...],
                'robot_obs': [Tensor(15), ...]
            }

            'demonstration_observations': [{observation}, ...],
            'demonstration_observation_num': number,
            'runtime_observation': observation
        """
        batch_size = ...
        seq_length = ...
        demonstration_images_batch = demonstration_images.view(batch_size * seq_length, 3, 200, 200)

        demonstration_images_batch = self.vision_encoder(demonstration_images_batch)

        demonstration_x = demonstration_images_batch.view(batch_size, seq_length, 64)

        # Apply dropout
        demonstration_x = self.dropout(demonstration_x)

        # Pack sequence for input to encoder LSTM
        demonstration_x = nn.utils.rnn.pack_padded_sequence(demonstration_x, demonstration_images_num.cpu(), batch_first=True)

        # Get the output from the encoder LSTM
        _encoder_outputs, (encoder_final_hidden, _encoder_final_cell) = self.lstm_encoder(demonstration_x)
        # encoder_final_hidden (batch, encoder_hidden_dim(64))

        ######################################## DECODER ########################################
        # Encode runtime image (batch, img_embed_dim(64))
        runtime_x = self.vision_encoder(runtime_image)

        # Apply dropout
        runtime_x = self.dropout(runtime_x)

        # Concatenate the encoder's final hidden state to the encoded runtime image
        # (batch, img_embed_dim(64) + encoder_hidden_dim(64))
        runtime_x = torch.cat(
            [runtime_x, encoder_final_hidden.squeeze()], #.unsqueeze(0)], # TODO: At evaluation, getting dim errors, fixed with this weird trick
            dim=1,
        )
        # (batch, 128)

        # Final output from the LSTM decoder (predicted action when given input runtime_image)
        output, _ = self.lstm_decoder(
            runtime_x.unsqueeze(0).transpose(0, 1),  # convert to shape `(batch, tgt_len(1), dim(128))`
            (encoder_final_hidden,  # hidden
            torch.zeros_like(encoder_final_hidden))  # cell
        )
        # output (batch, tgt_len(1), decoder_hidden_size(64))

        # convert to shape `(tgt_len(1), batch, decoder_hidden_size(64))`
        x = output.transpose(0, 1)

        # squeeze to shape (batch, decoder_hidden_size(64))
        x = x.squeeze(0)

        # Project the outputs to the action space (7,).
        x = self.output_projection(x)

        # Ensures actions are in range (-1, 1)
        x = self.normalize(x)

        # Return the predicted action and ``None`` for the attention weights
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(
        self,
        train_batch: Tuple[Dict, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """_summary_

        Args:
                train_batch (tuple): ({'rgb_static': ..., 'rgb_gripper': ...}, robot_obs)
                batch_idx (int): _description_

        Returns:
                torch.Tensor: _description_
        """
        # demonstration_images (list of tensor images), demonstration_images_num (number of demonstrations), runtime_image (single image to "decode")

        x, y = train_batch
        demonstration_images = x.get('demonstration_images')
        demonstration_images_num = x.get('demonstration_images_num')
        runtime_image = x.get('runtime_image')

        y_hat = self(demonstration_images, demonstration_images_num, runtime_image)
        # print(y_hat)
        # print(y)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(
        self,
        train_batch: Tuple[Dict, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """_summary_

        Args:
                train_batch (tuple): ({'rgb_static': ..., 'rgb_gripper': ...}, robot_obs)
                batch_idx (int): _description_

        Returns:
                torch.Tensor: _description_
        """
        # demonstration_images (list of tensor images), demonstration_images_num (number of demonstrations), runtime_image (single image to "decode")

        x, y = train_batch
        demonstration_images = x.get('demonstration_images')
        demonstration_images_num = x.get('demonstration_images_num')
        runtime_image = x.get('runtime_image')

        y_hat = self(demonstration_images, demonstration_images_num, runtime_image)
        # print(y_hat)
        # print(y)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss


#############################

#############################

#############################

#############################


class LSTMDecoder(nn.Module):
    def __init__(
        self, args, dictionary, encoder_hidden_dim=64, embed_dim=64, hidden_dim=64, dropout=0.1,
    ):
        super().__init__(dictionary)
        self.args = args

        # encoder for images (B, 3, 200, 200) -> (B, embed_dim(64))
        self.vision_encoder = VisionNetwork(200, 200, 'LeakyReLU', 0.0, True, embed_dim, 3)

        self.dropout = nn.Dropout(p=dropout)

        # We'll use a single-layer, unidirectional LSTM for simplicity.
        # (B, embed_dim(64) + encoder_embed_dim(64)) -> (B, hidden_dim(64))
        self.lstm = nn.LSTM(
            input_size=embed_dim + encoder_hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Define the output projection to our action space (B, hidden_dim(64)) -> (B, action_space(7))
        self.output_projection = nn.Linear(hidden_dim, 7)

        # Normalize to range (-1, 1) for actions
        self.normalize = nn.Sigmoid()

    ########################################################################################################

    # During training Decoders are expected to take the entire target sequence
    # (shifted right by one position) and produce logits over the vocabulary.
    # The *prev_output_tokens* tensor begins with the end-of-sentence symbol,
    # ``dictionary.eos()``, followed by the target sequence.

    def forward(self, prev_output_tokens, encoder_out):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
        """
        batch, tgt_len = prev_output_tokens.size()

        # Extract the final hidden state from the Encoder.
        final_encoder_hidden = encoder_out['final_hidden']

        # Embed the target sequence, which has been shifted right by one
        # position and now starts with the end-of-sentence symbol.
        x = self.embed_tokens(prev_output_tokens)

        # Apply dropout.
        x = self.dropout(x)

        # Concatenate the Encoder's final hidden state to *every* embedded
        # target token.
        x = torch.cat(
            [x, final_encoder_hidden.unsqueeze(1).expand(batch, tgt_len, -1)],
            dim=2,
        )

        # Using PackedSequence objects in the Decoder is harder than in the
        # Encoder, since the targets are not sorted in descending length order,
        # which is a requirement of ``pack_padded_sequence()``. Instead we'll
        # feed nn.LSTM directly.
        initial_state = (
            final_encoder_hidden.unsqueeze(0),  # hidden
            torch.zeros_like(final_encoder_hidden).unsqueeze(0),  # cell
        )
        output, _ = self.lstm(
            x.transpose(0, 1),  # convert to shape `(tgt_len, batch, dim)`
            initial_state,
        )
        x = output.transpose(0, 1)  # convert to shape `(batch, tgt_len, hidden)`

        # Project the outputs to the size of the vocabulary.
        x = self.output_projection(x)

        # Return the logits and ``None`` for the attention weights
        return x, None
