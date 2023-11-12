#
#
# import torch
# import torch.nn as nn
#
# class base_Model(nn.Module):
#     def __init__(self, configs):
#         super(base_Model, self).__init__()
#
#         # Constants for convolutional parameters
#         in_channels = configs.input_channels
#         out_channels = [32, 64, configs.final_out_channels]
#         kernel_sizes = [configs.kernel_size, 8, 8]
#         strides = [configs.stride, 1, 1]
#         paddings = [configs.kernel_size // 2, 4, 4]
#
#         # Create a list to store convolutional blocks
#         conv_blocks = []
#
#         for i in range(3):
#             conv_block = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels[i], kernel_size=kernel_sizes[i],
#                           stride=strides[i], padding=paddings[i], bias=False),
#                 nn.BatchNorm1d(out_channels[i]),
#                 nn.ReLU(),
#                 nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#             )
#             conv_blocks.append(conv_block)
#             in_channels = out_channels[i]
#
#         self.conv_blocks = nn.ModuleList(conv_blocks)
#
#         # Linear layer for classification
#         model_output_dim = configs.features_len
#         self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)
#
#     def forward(self, x_in):
#         x = x_in
#         for conv_block in self.conv_blocks:
#             x = conv_block(x)
#
#         x_flat = x.view(x.size(0), -1)
#         logits = self.logits(x_flat)
#         return logits, x
#
import torch
import torch.nn as nn
import torch.optim as optim

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        # Constants for convolutional parameters
        in_channels = configs.input_channels
        out_channels = [32, 64, configs.final_out_channels]
        kernel_sizes = [configs.kernel_size, 8, 8]
        strides = [configs.stride, 1, 1]
        paddings = [configs.kernel_size // 2, 4, 4]

        # Create a list to store convolutional blocks
        conv_blocks = []

        for i in range(3):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels[i], kernel_size=kernel_sizes[i],
                          stride=strides[i], padding=paddings[i], bias=False),
                nn.BatchNorm1d(out_channels[i]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            )
            conv_blocks.append(conv_block)
            in_channels = out_channels[i]

        self.conv_blocks = nn.ModuleList(conv_blocks)

        # Linear layer for classification
        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = x_in
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x_flat = x.view(x.size(0), -1)
        logits = self.logits(x_flat)
        return logits, x

