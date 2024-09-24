import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

import utils.geom
import utils.vox
import utils.misc
import utils.basic


from torchvision.models.resnet import resnet18
from efficientnet_pytorch import EfficientNet

EPS = 1e-4

from functools import partial

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False), # performs bilinear interpolation to upsample the feature map
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip

class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()
        """
            Decoder uses an U-Net alike architecture
        """

        """
            zero_init_residual will initialize the final layer of the BN in the residual block to 0. 
            during the initial stages of training, this allows the output of the residual block to be close to the input.
            this will ultimately help stabilize the model training.
        """
        backbone = resnet18(pretrained=False, zero_init_residual=True) 
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1 # does not effect the shape of the output tensor
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.predict_future_flow = predict_future_flow

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

    def forward(self, x, bev_flip_indices=None):
        
        b, c, h, w = x.shape # (B, 128, 200, 200)

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x) # keeps the shape same
        skip_x['2'] = x
        x = self.layer2(x) # (B, 64, 100, 100) ----> (B, 128, 50, 50)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        """
            1. Upsamples x
            2. Adds skip_x['3']
        """
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2]) # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])


        """x is passed on to different types of heads"""
        feat_output = self.feat_head(x)
        segmentation_output = self.segmentation_head(x) # CNN-based segmentatoin head
        instance_center_output = self.instance_center_head(x) # (B, 1, 200, 200)
        instance_offset_output = self.instance_offset_head(x) # (B, 2, 200, 200)
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None


        """
            Return data using a dictionary type
                - view(): this function is used to reshape tensors
                - *: unpacking operator
        """

        return {
            'raw_feat': x,
            'feat': feat_output.view(b, *feat_output.shape[1:]),
            'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
        }

import torchvision
class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=True)
        """excluding the last 4 layers"""
        self.backbone = nn.Sequential(*list(resnet.children())[:-4]) 
        self.layer3 = resnet.layer3
        
        """
        the operation below does two things
        1. changes the depth(channel) of the feature map while keeping the spatial resolution same
        2. furthur feature encoding
        """
        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)

        """concat two feature maps"""
        x = self.upsampling_layer(x2, x1) 

        x = self.depth_layer(x)

        return x

class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        # print('input_1', input_1.shape)
        # print('input_2', input_2.shape)
        x = self.upsampling_layer(input_1, input_2)
        # print('x', x.shape)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head
        return x

class Segnet(nn.Module):
    """
        Actual implementation of the Simple-BEV class
    """
    def __init__(self, Z, Y, X, vox_util=None, 
                 use_radar=False,
                 use_lidar=False,
                 use_metaradar=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 encoder_type="res101"):
        super(Segnet, self).__init__() # this code will invoke the init() of the super class
        assert (encoder_type in ["res101", "res50", "effb0", "effb4"])

        # You need to use `self.` like this so that other functions or classes can access it.
        self.Z, self.Y, self.X = Z, Y, X
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.use_metaradar = use_metaradar
        self.do_rgbcompress = do_rgbcompress   
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        
        """
        These are the mean and standard deviation for the ImageNet
        These values are important, because they will help match the distribution of nuScenes to the normalized ImagenNet's data distribution
        """
        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda() # `.cuda()` is similar to `.to(device)` when device is gpu.
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()
        
        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        """ResNet-101 is the default version"""
        if encoder_type == "res101": 
            self.encoder = Encoder_res101(feat2d_dim) # has 37,038,272 number of parameters
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == "effb0":
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # BEV compressor
        if self.use_radar:
            if self.use_metaradar:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y + 16*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y+1, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
        elif self.use_lidar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim*Y+Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else: # only RGB
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential( # this compressor will have 1,179,648 parameters
                    nn.Conv2d(feat2d_dim*Y, feat2d_dim*1, kernel_size=3, padding=1, stride=1, bias=False), # this conv operation will keep the spatial resolution same. it only reduces the height to one
                    nn.InstanceNorm2d(latent_dim), 
                    nn.GELU(),  
                )
            else:
                # use simple sum
                pass

        # Decoder
        """
        the decoder does the following things
        1. process multi-modal data using a u-net structure
        2. multiple heads to create different types of output
        """

        self.decoder = Decoder( # The decoder has 3,830,788
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False
        )

        # Loss
        """
        As mentioned in the paper, there are 3 losses.
        SegNet uses a multi-task loss which is composed of the following 3 losses
            - 1 main loss
            - 2 auxilary losses
        """
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True) # main: sem seg loss
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True) # aux: centerness loss
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True) # aux: offset loss
            
        # set_bn_momentum(self, 0.1)
        if vox_util is not None:
            self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False) # ego-car-base 3d space coordinates (100m, 100m, 10m)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z, Y, X, assert_cube=False) # discretized version into (200, 200, 8)
        else:
            self.xyz_camA = None
        
    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs
            - input RGB images
            - shape: (B,S,C,H,W)
        pix_T_cams
            - camera intrinsic matrix
            - shape: (B,S,4,4)
        cam0_T_camXs
            - extrinsic matrix (rotation, translatoin)
            - shape: (B,S,4,4)
        vox_util: vox util object
        rad_occ_mem0:
            - None when use_radar = False, use_lidar = False
            - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
            - (B, 1, Z, Y, X) when use_lidar = True
        '''
        
        B, S, C, H, W = rgb_camXs.shape # (batch, 6, 3, H, W)
        assert(C==3) # assert that there is an error with the channel of the images.
        
        """
            Reshaping Tensors
                Q: What is happening here?
                A: Combine the batch and camera dimensions.
                Purpose: for efficient batch processing.
            
            lambda functions are used for reshaping the tensors
        """
        
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_camXs_ = __p(rgb_camXs) # (B, 6, 3, H, W) ----> (Bx6, 3, H, W)
        pix_T_cams_ = __p(pix_T_cams) # (B, 6, 4, 4) ----> (BX6, 4, 4)
        cam0_T_camXs_ = __p(cam0_T_camXs) # (B, 6, 4, 4) ----> (BX6, 4, 4)
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_) # computes the inverse of a matrix in a numerically stable manner. camXs_T_cam0_ still has the same shape

        """
            RGB Encoder
                - Input shape:  (Bx6, 3, H, W)
                - Output shape: (Bx6, latent_dim, Hf, Wf)
        """
        device = rgb_camXs_.device
        rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device) # normalize the input data
        if self.rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
            rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])
        feat_camXs_ = self.encoder(rgb_camXs_) # feat_camXs.shape = (Bx6, latent_dim, Hf, Wf)
        if self.rand_flip:
            feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_camXs_.shape

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X

        """
            The resolution of the feature map has decreased because the image passed through a bunch of CNN layers.
            Therefore, we need to adjust the intrinsic matrix.
            Intrinsic matrix is composed of Focal length and center coordinate.
            So, we have to divide those values by the reduction rate.
        """
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy) # scale_intrinsics() function performs this complicated job for us.


        """
            Since there are `Bx6` number of feature maps in a batch,
            we need to expand set of 3D points for each feature map in a batch.
        """
        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*S,1,1) # dimension is expanded: (1, 320000, 3) ----> (Bx6, 320000, 3)
        else:
            xyz_camA = None

        """
            Feature Map to 3D Space projection
            
                Location of `unproject_image_to_mem()`
                    ----> utils/vox.py/unproject_image_to_mem()

                grid_sample() does the bilinear interpolation
        """
        feat_mems_ = vox_util.unproject_image_to_mem(
            feat_camXs_,
            utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_), # combine two transformations (intrinsic @ extrinsic)
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)
        feat_mems = __u(feat_mems_) # (24, 128, 200, 8, 200) ----> (4, 6, 128, 200, 8, 200) : (B, S, C, Z, Y, X)
        
        """
            1. absolute value
            2. check if bigger than 0
            3. Convert to float
                False ----> 0.0
                True  ----> 1.0
        """
        mask_mems = (torch.abs(feat_mems) > 0).float() # create valid volume mask

        """By executing the code below, the 6 3D features are finally combined into one 3D feature"""
        feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # Output shape: (B, C, 200, 8, 200)

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        """
            BEV Compressing
            At the end feat_bev will be created which represents the combined bev feature
        """
        if self.use_radar:
            assert(rad_occ_mem0 is not None)
            if not self.use_metaradar:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                rad_bev = torch.sum(rad_occ_mem0, 3).clamp(0,1) # squish the vertical dim
                feat_bev_ = torch.cat([feat_bev_, rad_bev], dim=1)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16*Y, Z, X)
                feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
                feat_bev = self.bev_compressor(feat_bev_)
        elif self.use_lidar:
            assert(rad_occ_mem0 is not None)
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
            feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
            feat_bev = self.bev_compressor(feat_bev_)
        else: # RGB only
            if self.do_rgbcompress:
                """
                    1. permute() manipulates the dimensions
                        (B, 128, 200, 8, 200) ----> (B, 128, 8, 200, 200)
                    2. reshape reshapes the tensor
                        (B, 128, 8, 200, 200) ----> (B, 1024, 200, 200)
                """
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X) 
                feat_bev = self.bev_compressor(feat_bev_) # CNN-base BEV Compressor
            else:
                feat_bev = torch.sum(feat_mem, dim=3) # the simplest method

        """
            Decoder: returns a dictionary typed output
        """
        out_dict = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        raw_e = out_dict['raw_feat']
        feat_e = out_dict['feat']
        seg_e = out_dict['segmentation']
        center_e = out_dict['instance_center']
        offset_e = out_dict['instance_offset']

        return raw_e, feat_e, seg_e, center_e, offset_e