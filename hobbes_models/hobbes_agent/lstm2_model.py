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


class HobbesActionDecoderLSTM(pl.LightningModule):
    def __init__(
        self, input_dim=256, hidden_dim=256, output_dim=7+1, dropout=0.1,
    ):
        super().__init__()

        self.lstm_decoder = nn.LSTM(input_size=input_dim,
                                    hidden_size=hidden_dim,
                                    proj_size=output_dim,
                                    num_layers=1,
                                    batch_first=True)

        self.dropout = nn.Dropout(p=dropout)

        # normalize output to range (-1,1)
        self.normalize = nn.Tanh()

    def forward(self, input_embeddings, num_embeddings):
        """_summary_

        Args:
            input_embeddings Tensor(N, L, input_dim): padded input embeddings for all frames
            num_embeddings (number): number of frames passed in for each batch
        """

        # input_embeddings: (N, L, input_dim)

        input_embeddings = self.dropout(input_embeddings)

        # initial hidden state and cell state to use
        # h_0 = torch.randn(2, 3, 20)
        # c_0 = torch.randn(2, 3, 20)

        packed_input_embeddings = nn.utils.rnn.pack_padded_sequence(input_embeddings, num_embeddings, batch_first=True)

        packed_output, (h_n, c_n) = self.lstm_decoder(packed_input_embeddings) #, (h_0, c_0))

        output = nn.utils.rnn.pad_packed_sequence(packed_output, num_embeddings, batch_first=True)

        # output: (N, L, output_dim)
        # h_n: final hidden state
        # c_n: final cell state

        output = self.normalize(output)

        return output
    
    def training_step(self, train_batch, train_batch_lengths):
        """_summary_

        Args:
                train_batch (tuple): ( (N, L, input_dim), (N, L, output_dim) )
                train_batch_lengths (list(int)): L for each batch

        Returns:
                torch.Tensor: _description_
        """
        x, y = train_batch
        y_hat = self(x, train_batch_lengths)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class HobbesDecoderWrapper(pl.LightningModule):
    def __init__(
        self, static_camera_embed_dim=64, gripper_camera_embed_dim=32, robot_obs_dim=15, hidden_dim=256, dropout=0.1,
    ):
        super().__init__()

        self.static_camera_encoder = VisionNetwork(200, 200, 'LeakyReLU', 0.0, True, static_camera_embed_dim, 3)

        self.gripper_camera_encoder = VisionNetwork(84, 84, 'LeakyReLU', 0.0, True, static_camera_embed_dim, 3)

        self.robot_obs_encoder = nn.Identity()

        self.ff = nn.Sequential(
            nn.Linear(static_camera_embed_dim+gripper_camera_embed_dim+robot_obs_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
        )

        self.action_lstm = HobbesActionDecoderLSTM(input_size=static_camera_embed_dim + gripper_camera_embed_dim + robot_obs_dim,
                                                   hidden_size=hidden_dim,
                                                   num_layers=1,
                                                   batch_first=True)

    def forward(self, static_camera_obs, gripper_camera_obs, robot_obs):

        static_camera_embeddings = self.static_camera_encoder(static_camera_obs)
        gripper_camera_embeddings = self.gripper_camera_encoder(gripper_camera_obs)
        robot_embeddings = self.robot_obs_encoder(robot_obs)
        all_embeddings = torch.cat([static_camera_embeddings, gripper_camera_embeddings, robot_embeddings], dim=-1)

        input_embeddings = self.ff(all_embeddings)

        self.action_lstm(input_embeddings, 0)

        

    def training_step(self, train_batch, train_batch_lengths):
        """_summary_

        Args:
                train_batch (tuple): ( (N, L, input_dim), (N, L, output_dim) )
                train_batch_lengths (list(int)): L for each batch

        Returns:
                torch.Tensor: _description_

        train_batch (tuple): (state, target_actions)

        state (dict): {'rgb_static': list(images), 'rgb_gripper': list(images), 'robot_obs': list(robot_obs)}
        """
        x, y = train_batch
        
        observations = x.get('state_observations')
        num_observations = x.get('state_observations_num')

        static_obs = observations.get('rgb_static')
        gripper_obs = observations.get('rgb_gripper')
        robot_obs = observations.get('robot_obs')

        y_hat = self(static_obs, gripper_obs, robot_obs, num_observations)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
#################### 
        

# Encodes static camera, gripper camera, and robot proprioceptive information into a single vector
class HobbesStateEncoderLSTM(pl.LightningModule):
    def __init__(
        self, static_camera_embed_dim=64, gripper_camera_embed_dim=32, robot_obs_dim=15, hidden_dim=256, dropout=0.1,
    ):
        super().__init__()

        self.static_camera_encoder = VisionNetwork(200, 200, 'LeakyReLU', 0.0, True, static_camera_embed_dim, 3)

        self.gripper_camera_encoder = VisionNetwork(84, 84, 'LeakyReLU', 0.0, True, static_camera_embed_dim, 3)

        self.lstm_encoder = nn.LSTM(input_size=static_camera_embed_dim+gripper_camera_embed_dim+robot_obs_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=1,
                                    batch_first=True)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_embeddings, num_embeddings):
        """_summary_

        Args:
            input_embeddings Tensor(N, L, input_dim): padded input embeddings for all frames
            num_embeddings (number): number of frames passed in for each batch
        """

        # input_embeddings: (N, L, input_dim)

        input_embeddings = self.dropout(input_embeddings)

        # initial hidden state and cell state to use
        # h_0 = torch.randn(2, 3, 20)
        # c_0 = torch.randn(2, 3, 20)

        packed_input_embeddings = nn.utils.rnn.pack_padded_sequence(input_embeddings, num_embeddings, batch_first=True)

        packed_output, (h_n, c_n) = self.lstm_decoder(packed_input_embeddings) #, (h_0, c_0))

        output = nn.utils.rnn.pad_packed_sequence(packed_output, num_embeddings, batch_first=True)

        # output: (N, L, output_dim)
        # h_n: final hidden state
        # c_n: final cell state

        output = self.normalize(output)

        return output
    
    def training_step(self, train_batch, train_batch_lengths):
        """_summary_

        Args:
                train_batch (tuple): ( (N, L, input_dim), (N, L, output_dim) )
                train_batch_lengths (list(int)): L for each batch

        Returns:
                torch.Tensor: _description_
        """
        x, y = train_batch
        y_hat = self(x, train_batch_lengths)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
        













#############################################################################################################################
### BELOW THIS LINE IS OUTDATED





class HobbesEncoderLSTM(pl.LightningModule):
    def __init__(
        self, static_img_embed_dim=64, gripper_img_embed_dim=32, encoder_hidden_dim=64, dropout=0.1,
    ):
        super().__init__()

        # encoder for images (batch, 3, 200, 200) -> (batch, embed_dim(64))
        self.static_img_encoder = VisionNetwork(200, 200, 'LeakyReLU', 0.0, True, static_img_embed_dim, 3)

        self.gripper_img_encoder = VisionNetwork(84, 84, 'LeakyReLU', 0.0, True, gripper_img_embed_dim, 3)

        self.proprio_encoder = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)

        # LSTM https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # (batch, img_embed_dim(96) + robot_obs(15)) -> (batch, encoder_hidden_dim(64))
        self.lstm_encoder = nn.LSTM(input_size=static_img_embed_dim + gripper_img_embed_dim + 15,
                                    hidden_size=encoder_hidden_dim,
                                    num_layers=1,
                                    batch_first=True)

    # Assuming we are given a sequence of demonstration images, and one single image from runtime
    def forward(self, demonstration_states, num_demonstration_states):
        ######################################## ENCODER ########################################
        # Encode demonstration input images (batch, img_embed_dim(64))
        # Confound sequence and batch for Conv2D https://discuss.pytorch.org/t/processing-sequence-of-images-with-conv2d/37494

        """ TODO: make this work probably.... all the data is laid out already
            
            demonstration_states: {
                'rgb_static': [Tensor(3,200,200), ...],
                'rgb_gripper': [Tensor(3,84,84), ...],
                'robot_obs': [Tensor(15), ...]
            }

            number of demonstrations
            num_demonstration_states: number
            
        """
        
        static_images = demonstration_states['rgb_static']

        batch_size = static_images.shape[0]
        seq_length = static_images.shape[1]

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
