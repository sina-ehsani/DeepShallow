import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSelfAttentionCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, height=27, width=8, bias=True):
        super(ConvSelfAttentionCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.height = height
        self.width = width

        self.query_conv = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.key_conv = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.value_conv = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.concat_conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.layer_norm = nn.LayerNorm([hidden_dim, self.height, self.width])

    def forward(self, input_tensor):
        # input_tensor shape: [batch_size, window_size, channels, height, width]
        batch_size, seq_len, _, height, width = input_tensor.size()

        # reshape to: [batch_size * window_size, channels, height, width]
        input_tensor = input_tensor.view(batch_size * seq_len, -1, height, width)

        # project the input tensor to query, key, and value tensors
        proj_query = (
            self.query_conv(input_tensor).view(batch_size, seq_len, -1, width * height).permute(0, 2, 1, 3)
        )  # B x C x seq_len x (H*W)
        proj_key = (
            self.key_conv(input_tensor).view(batch_size, seq_len, -1, width * height).permute(0, 2, 3, 1)
        )  # B x C x (H*W) x seq_len
        proj_value = (
            self.value_conv(input_tensor).view(batch_size, seq_len, -1, width * height).permute(0, 2, 1, 3)
        )  # B x C x seq_len x (H*W)

        # calculate the attention scores
        energy = torch.matmul(proj_query, proj_key)  # B x C x seq_len x seq_len
        attention = F.softmax(energy, dim=-1)  # softmax over the last dimension

        # apply the attention scores to the value tensor
        out = torch.matmul(attention, proj_value)  # B x C x seq_len x (H*W)
        out = out.view(
            batch_size * seq_len, -1, height, width
        )  # reshape to: [batch_size * window_size, channels, height, width]
        # out = out.view(batch_size, seq_len, -1, height, width)  # reshape to the original size

        concat_out = self.concat_conv(torch.cat((input_tensor, out), dim=1))
        concat_out = concat_out.view(batch_size, seq_len, -1, height, width)  # reshape to the original size
        concat_out = self.layer_norm(concat_out)

        # print(f"ConvSelfAttentionCell input: {input_tensor.shape} \n proj_query : {proj_query.shape} , proj_key : {proj_key.shape} , proj_value: {proj_value.shape} ")
        # print(f"energy: {energy.shape}, attention : {attention.shape} \n out : {out.shape} , concat_out : {concat_out.shape} ")

        return concat_out


class ConvSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels, tensor_size, window):
        super(ConvSelfAttention, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.height, self.width = tensor_size
        self.window = window

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvSelfAttentionCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    height=self.height,
                    width=self.width,
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

        self.conv1 = nn.Conv2d(in_channels=in_channels // 2, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, input_tensor, input_standalone):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        layer_output_list = [input_tensor]
        for layer_idx in range(self.num_layers):
            # print(f"input_tensor of {layer_idx} is  {input_tensor.shape}")
            h = self.cell_list[layer_idx](layer_output_list[-1])
            layer_output_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]

        # Now process the input_tensor through the ConvSelfAttention and the standalone input through a separate Conv2D
        conv_out = self.conv1(layer_output_list[0][:, -1, :, :, :])
        standalone_out = self.conv2(input_standalone)

        # Concatenate the outputs along the channel dimension
        combined_out = torch.cat((conv_out, standalone_out), dim=1)

        # Pass the concatenated output through another Conv2D to get the final output
        final_out = self.conv3(combined_out)

        return final_out

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
