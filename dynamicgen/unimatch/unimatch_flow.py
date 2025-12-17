import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path

from unimatch.unimatch import UniMatch
from unimatch.utils.utils import InputPadder


class FlowOcclusionProcessor(torch.nn.Module):
    """
    基于UniMatch的光流Processing器
    """
    
    def __init__(self, 
                 flow_model="unimatch", 
                 pair_mode="sequential", 
                 use_existing_flow=False,
                 model_path=None):
        """
        初始化光流Processing器
        
        Args:
            flow_model (str): 光流模型类型
            pair_mode (str): 帧对模式
            use_existing_flow (bool): 是否使用现有光流
            model_path (str): 模型权重路径
        """
        super(FlowOcclusionProcessor, self).__init__()
        
        self.flow_model = flow_model
        self.pair_mode = pair_mode
        self.use_existing_flow = use_existing_flow
        
        # 默认模型路径
        self.model_path = model_path
        
        # 初始化UniMatch模型
        self.model = UniMatch(
            num_scales=2,
            feature_channels=128,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task='flow'
        )
        
        # Loading预训练权重
        self.load_pretrained_weights()
        
        logging.info(f"初始化FlowOcclusionProcessor，模型: {flow_model}")
    
    def load_pretrained_weights(self):
        """Loading预训练权重"""
        try:
            if Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=True)
                self.model.load_state_dict(checkpoint['model'], strict=False)
                logging.info(f"SuccessLoading预训练权重: {self.model_path}")
            else:
                logging.warning(f"预训练权重文件不存在: {self.model_path}")
        except Exception as e:
            logging.error(f"Loading预训练权重Failed: {e}")
    
    def forward(self, img_pair):
        """
        前向传播计算光流
        
        Args:
            img_pair (torch.Tensor): 输入图像对，形状为 [B, 2, 3, H, W]
            
        Returns:
            torch.Tensor: 输出结果，包含光流信息
        """
        B, T, C, H, W = img_pair.shape
        assert T == 2, f"期望输入2帧图像，但得到 {T} 帧"
        
        # 分离两帧图像
        img0 = img_pair[:, 0]  # [B, 3, H, W]
        img1 = img_pair[:, 1]  # [B, 3, H, W]
        
        # 输入填充以确保尺寸可被32整除
        padder = InputPadder(img0.shape, padding_factor=32)
        img0, img1 = padder.pad(img0, img1)
        
        # 使用UniMatch计算光流
        with torch.no_grad():
            results = self.model(img0, img1,
                               attn_type='swin',
                               attn_splits_list=[2, 8],
                               corr_radius_list=[-1, 4],
                               prop_radius_list=[-1, 1],
                               num_reg_refine=6)
        
        # from字典中extraction光流预测
        flow_preds = results['flow_preds']
        
        # 移除填充
        # UniMatch返回一个光流预测列表，最后一个是最终结果
        flow_up = padder.unpad(flow_preds[-1])
        
        # 构造输出格式以匹配原始代码期望
        # 原始代码期望格式: [B, T, 5, H, W]，其中5个通道包含 [img0, img1, flow_x, flow_y, occlusion]
        B, _, H_out, W_out = flow_up.shape
        
        # 创建输出张量
        output = torch.zeros(B, 1, 5, H_out, W_out, device=img_pair.device, dtype=img_pair.dtype)
        
        # 填充图像信息（降采样到输出尺寸）
        img0_resized = F.interpolate(img0, (H_out, W_out), mode='bilinear', align_corners=False)
        img1_resized = F.interpolate(img1, (H_out, W_out), mode='bilinear', align_corners=False)
        
        # 将图像from[0,1]范围调整到适当的范围
        output[:, 0, 0] = img0_resized[:, 0]  # R通道
        output[:, 0, 1] = img0_resized[:, 1]  # G通道  
        output[:, 0, 2] = img0_resized[:, 2]  # B通道
        
        # 填充光流信息
        output[:, 0, 3] = flow_up[:, 0]  # flow_x
        output[:, 0, 4] = flow_up[:, 1]  # flow_y
        
        return output
    
    def to(self, device):
        """移动模型到指定设备"""
        self.model = self.model.to(device)
        return super().to(device)
    
    def parameters(self):
        """返回模型参数"""
        return self.model.parameters()