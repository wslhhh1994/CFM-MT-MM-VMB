
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

        
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化 (B, C, L) -> (B, C, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, channels, length)
        b, c, l = x.size()
        y = self.avg_pool(x).view(b, c)  # 压缩时间维度
        y = self.fc(y).view(b, c, 1)     # 生成通道权重
        return x * y.expand_as(x)        # 权重与输入相乘        

class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        attention_weights = self.attention(x)  # (B, 1, L)
        return x * attention_weights    
    
    
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_weights = self.conv1(x)
        attention_weights = self.sigmoid(attention_weights)
        return x * attention_weights.expand_as(x)

class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.temporal_attention = TemporalAttention(in_channels)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        # Parallel Attention Mechanism
        x_channel = self.channel_attention(x)
        x_temporal = self.temporal_attention(x)
        
        # Combine all attentions
        x = x_channel * x_temporal  # Element-wise multiplication
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        return x

class SelfAttention(nn.Module):
    def __init__(self, channels, size,model_size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = model_size
#         self.size_x = int(model_size/20)
        self.size_x = int(model_size/8)#这里和nchannel有关  thie para* n channel=model size  #size x 实际上就是 n channel  8还是 16？

        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # print(self.channels, self.size_x, self.size)

        x = x.view(-1, self.channels, self.size_x * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size_x, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        # print(x.shape)
        # print(t.shape)
        # print(y.shape)

        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        # print(skip_x.shape)
        # print(x.shape)

        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
#         print(emb.shape,'emb')
#         print(x.shape,'x')

        return x + emb


class Down0(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        # print(x.shape)
        # print(t.shape)
        # print(y.shape)

        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x# + emb


class Up0(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        # print(skip_x.shape)
        # print(x.shape)

        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
#         print(emb.shape,'emb')
#         print(x.shape,'x')

        return x #+ emb
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, model_size=64,device="cuda"):#model_size 就是模型的深度 然后SelfAttention也需要改
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down0(16, 32)
        self.sa1 = SelfAttention(32, 8,int(model_size/2))
        self.down2 = Down0(32, 64)
        self.sa2 = SelfAttention(64, 4,int(model_size/4))
        self.down3 = Down0(64, 64)
        self.sa3 = SelfAttention(64, 2,int(model_size/8))

        self.bot1 = DoubleConv(64, 128)
        self.bot2 = DoubleConv(128, 128)
        self.bot3 = DoubleConv(128, 64)

        self.up1 = Up0(128, 32)
        self.sa4 = SelfAttention(32, 4,int(model_size/4))
        self.up2 = Up0(64, 16)
        self.sa5 = SelfAttention(16, 8,int(model_size/2))
        self.up3 = Up0(32, 16)
        self.sa6 = SelfAttention(16, 16,int(model_size/1))
        self.outc = nn.Conv2d(16, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)


#         print(x.shape,'x')
        x1 = self.inc(x)
#         print(x1.shape,'x1')

        x2 = self.down1(x1, t)
#         print(x2.shape,'x2')

        x2 = self.sa1(x2)
#         print(x2.shape,'x2')

        x3 = self.down2(x2, t)
#         print(x3.shape,'x3')
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
#         print(x4.shape,'x4')
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
#         print(x.shape,'x')
#         print(x2.shape,'x2')

        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, model_size=64,number_traces=32,device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.number_traces=number_traces
        self.model_size=model_size

        
        

#         self.conv1d_1 = nn.Conv1d(in_channels=self.number_traces,
#                                   out_channels=64,
#                                   kernel_size=21, padding=10)
#         self.conv1d_2 = nn.Conv1d(in_channels=64,
#                                   out_channels=32,
#                                   kernel_size=21, padding=10)
#         self.conv1d_3 = nn.Conv1d(in_channels=32,
#                                   out_channels=8,
#                                   kernel_size=21, padding=10)
#         self.conv1d_4 = nn.Conv1d(in_channels=8,
#                                   out_channels=1,
#                                   kernel_size=21, padding=10)

# 
        self.conv1d_1 = nn.Conv1d(in_channels=self.number_traces,
                                  out_channels=64,
                                  kernel_size=21, padding=10)
        self.conv1d_2 = nn.Conv1d(in_channels=64,
                                  out_channels=1,
                                  kernel_size=21, padding=10)

        self.conv1d_11 = nn.Conv1d(in_channels=self.number_traces,
                                  out_channels=64,
                                  kernel_size=21, padding=10)

        self.conv1d_22 = nn.Conv1d(in_channels=64,
                                  out_channels=1,
                                  kernel_size=21, padding=10)
        self.cat_3_condition=torch.cat
        
        self.conv1d_33 = nn.Conv1d(in_channels=3,
                                  out_channels=1,
                                  kernel_size=3, padding=1)
        # self.inc = DoubleConv(c_in, 64)
        # self.down1 = Down(64, 128)
        # self.sa1 = SelfAttention(128, 32,int(model_size/2))
        # self.down2 = Down(128, 256)
        # self.sa2 = SelfAttention(256, 16,int(model_size/4))
        # self.down3 = Down(256, 256)
        # self.sa3 = SelfAttention(256, 8,int(model_size/8))
        #
        #
        #
        # self.bot1 = DoubleConv(256, 512)
        # self.bot2 = DoubleConv(512, 512)
        # self.bot3 = DoubleConv(512, 256)
        #
        # self.up1 = Up(512, 128)
        # self.sa4 = SelfAttention(128, 16,int(model_size/4))
        # self.up2 = Up(256, 64)
        # self.sa5 = SelfAttention(64, 32,int(model_size/2))
        # self.up3 = Up(128, 64)
        # self.sa6 = SelfAttention(64, 64,int(model_size/1))
        # self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down(16, 32)
        self.sa1 = SelfAttention(32, 8,int(model_size/2))
        self.down2 = Down(32, 64)
        self.sa2 = SelfAttention(64, 4,int(model_size/4))
        self.down3 = Down(64, 64)
        self.sa3 = SelfAttention(64, 2,int(model_size/8))

        self.bot1 = DoubleConv(64, 128)
        self.bot2 = DoubleConv(128, 128)
        self.bot3 = DoubleConv(128, 64)

        self.up1 = Up(128, 32)
        self.sa4 = SelfAttention(32, 4,int(model_size/4))
        self.up2 = Up(64, 16)
        self.sa5 = SelfAttention(16, 8,int(model_size/2))
        self.up3 = Up(32, 16)
        self.sa6 = SelfAttention(16, 16,int(model_size/1))
        self.outc = nn.Conv2d(16, c_out, kernel_size=1)

        self.channel_attentiony = CBAMBlock(3)
        self.channel_attentionz = CBAMBlock(3)
        self.channel_attentionw = CBAMBlock(3)
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t,y=None,z=None,w=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None and z is not None and w is not None:
#             print(y.shape)
#             y=self.channel_attentiony(y)
            y1=self.conv1d_1(y)
            # print(y1.shape)
            y4=self.conv1d_2(y1)


            
#             y4=y4.view(-1,self.time_dim)
#             t+=y4
#             z=self.channel_attentionz(z)

            z1=self.conv1d_11(z)
            # print(y1.shape)
            z4=self.conv1d_22(z1)
#             w=self.channel_attentionz(w)

            w1=self.conv1d_11(w)
            # print(y1.shape)
            w4=self.conv1d_22(w1)
            
#             print(y4.shape,'y4')
#             print(z4.shape,'z4')
#             print(w4.shape,'w4')

#             yz=self.cat_2_condition([y4,z4],axis=0)
#             yz=(y4+z4+w4)/3
#             yz=self.channel_attentiony(yz)
            wyz_concat = self.cat_3_condition((y4, z4, w4), dim=1)
#             print(wyz_concat.shape)

            wyz_concat=self.channel_attentiony(wyz_concat)
            wyz=self.conv1d_33(wyz_concat)
#             print(wyz.shape)
#             print(yz.shape,'yz')
            wyz=wyz.view(-1,self.time_dim)
#             print(yz.shape,'yz')
#             print(t.shape,'t')
            t+=wyz
            
            



#         print(t.shape,'t')
#         print(x.shape,"x")
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
#         print(x2.shape,'x2')

        x2 = self.sa1(x2)
#         print(x2.shape,'x2')

        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output




# class UNet_conditional(nn.Module):
#     def __init__(self, c_in=3, c_out=3, time_dim=256, model_size=160,number_traces=15,device="cuda"):
#         super().__init__()
#         self.device = device
#         self.time_dim = time_dim
#         self.number_traces=number_traces
#
#         self.conv1d_1 = nn.Conv1d(in_channels=self.number_traces,
#                                   out_channels=64,
#                                   kernel_size=21, padding=10)
#         self.conv1d_2 = nn.Conv1d(in_channels=64,
#                                   out_channels=32,
#                                   kernel_size=21, padding=10)
#         self.conv1d_3 = nn.Conv1d(in_channels=32,
#                                   out_channels=8,
#                                   kernel_size=21, padding=10)
#         self.conv1d_4 = nn.Conv1d(in_channels=8,
#                                   out_channels=1,
#                                   kernel_size=21, padding=10)
#
#
#
#         self.inc = DoubleConv(c_in, 64)
#         self.down1 = Down(64, 128)
#         self.sa1 = SelfAttention(128, 32,int(model_size/2))
#         self.down2 = Down(128, 256)
#         self.sa2 = SelfAttention(256, 16,int(model_size/4))
#         self.down3 = Down(256, 256)
#         self.sa3 = SelfAttention(256, 8,int(model_size/8))
#
#         self.bot1 = DoubleConv(256, 512)
#         self.bot2 = DoubleConv(512, 512)
#         self.bot3 = DoubleConv(512, 256)
#
#         self.up1 = Up(512, 128)
#         self.sa4 = SelfAttention(128, 16,int(model_size/4))
#         self.up2 = Up(256, 64)
#         self.sa5 = SelfAttention(64, 32,int(model_size/2))
#         self.up3 = Up(128, 64)
#         self.sa6 = SelfAttention(64, 64,int(model_size/1))
#         self.outc = nn.Conv2d(64, c_out, kernel_size=1)
#
#
#
#     def pos_encoding(self, t, channels):
#         inv_freq = 1.0 / (
#             10000
#             ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
#         )
#         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         return pos_enc
#
#     def forward(self, x, t,y):
#         t = t.unsqueeze(-1).type(torch.float)
#         t = self.pos_encoding(t, self.time_dim)
#
#         if y is not None:
#             # print(y.shape)
#             y1=self.conv1d_1(y)
#             # print(y1.shape)
#
#             y2=self.conv1d_2(y1)
#             # print(y2.shape)
#
#             y3=self.conv1d_3(y2)
#             # print(y3.shape)
#
#             y4=self.conv1d_4(y3)
#             y4=y4.view(-1,self.time_dim)
#             # print(t.shape)
#             # print(y4.shape)
#             t+=y4
#
#
#
#         # print(t.shape)
#         # print(x.shape)
#         x1 = self.inc(x)
#         x2 = self.down1(x1, t)
#         # print(x2.shape)
#
#         x2 = self.sa1(x2)
#         # print(x2.shape)
#
#         x3 = self.down2(x2, t)
#         x3 = self.sa2(x3)
#         x4 = self.down3(x3, t)
#         x4 = self.sa3(x4)
#
#         x4 = self.bot1(x4)
#         x4 = self.bot2(x4)
#         x4 = self.bot3(x4)
#
#         x = self.up1(x4, x3, t)
#         x = self.sa4(x)
#         x = self.up2(x, x2, t)
#         x = self.sa5(x)
#         x = self.up3(x, x1, t)
#         x = self.sa6(x)
#         output = self.outc(x)
#         return output




if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = UNet_conditional(num_classes=1, device="cuda")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)
