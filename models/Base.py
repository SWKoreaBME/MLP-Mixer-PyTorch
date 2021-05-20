import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, out_features):
        super(Mlp, self).__init__()

        self.fc1 = nn.Linear(in_features, out_features, bias=True)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class TokenMixer(nn.Module):
    def __init__(self):
        super(TokenMixer, self).__init__()

    def forward(self, x):
        pass


class ChannelMixer(nn.Module):
    def __init__(self):
        super(ChannelMixer, self).__init__()

    def forward(self, x):
        pass


class Mixer(nn.Module):
    """
        N: Num patches
        C: Channels
        B: Mini-batch size

        input shape: B x C x N
        output shape: B x C x N
    """
    def __init__(self, in_channels, num_patches):
        super(Mixer, self).__init__()

        self.ln1 = nn.LayerNorm(num_patches)
        self.ln2 = nn.LayerNorm(num_patches)

        self.in_channels = in_channels
        self.num_patches = in_channels

        # generate mlps according to number of patches
        # self.token_mixer_list = nn.ModuleList([Mlp(num_patches, num_patches) for _ in range(in_channels)])
        self.token_mixer_list = [Mlp(num_patches, num_patches) for _ in range(in_channels)]
        print(self.token_mixer_list)
        self.channel_mixer_list = nn.ModuleList([Mlp(in_channels, in_channels) for _ in range(num_patches)])

    def forward(self, x):
        x_token_identity = x
        x = self.ln1(x)
        x_t = torch.transpose(x, -1, -2)  # B X N X C

        print(x_t.size())

        x_token_mixed = torch.zeros_like(x_t)
        for channel_idx in range(self.in_channels):
            token_mixer = self.token_mixer_list(channel_idx)
            print(token_mixer)
            x_token_mixed[:, :, channel_idx] = token_mixer(x_t[:, :, channel_idx])

        x = x_token_identity + torch.transpose(x_token_mixed, -1, -2)  # B x C x N
        x_channel_identity = x

        x = self.ln2(x)
        x_channel_mixed = torch.zeros_like(x)
        for patch_idx in range(self.num_patches):
            channel_mixer = self.channel_mixer_list(patch_idx)
            x_channel_mixed[:, :, patch_idx] = channel_mixer(x[:, :, patch_idx])

        out = x_channel_mixed + x_channel_identity
        return out


if __name__ == '__main__':
    patch_size = (32, 32)
    num_patches = 9
    num_channels = 3
    batch_size = 4

    input_tokens = torch.randn((batch_size, num_channels, num_patches))  # B x C x N
    mixer = Mixer(num_channels, num_patches)

    mixer_out = mixer(input_tokens)

    print(input_tokens.size(), mixer_out.size())

    pass
