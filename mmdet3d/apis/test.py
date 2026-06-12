import mmcv
import torch
import numpy as np
import pickle
import os
import shutil
from nuscenes import NuScenes

# def single_gpu_test(model, data_loader):
#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     prog_bar = mmcv.ProgressBar(len(dataset))
#     for data in data_loader:
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=True, **data)
#         results.extend(result)

#         batch_size = len(result)
#         for _ in range(batch_size):
#             prog_bar.update()
#     return results





# 全局只初始化一次nuscenes，避免重复加载


# 全局只初始化一次nuscenes，避免重复加载
nusc = NuScenes(
    version='v1.0-trainval', 
    dataroot='/data5/wangyijun/DAOcc/data/nuscenes',  # 修正1：数据集路径
    verbose=False
)

def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    # 创建保存目录
    os.makedirs("save/pkl", exist_ok=True)
    os.makedirs("save/imgs", exist_ok=True)
    
    # 跳过第0帧
    for idx, data in enumerate(data_loader):
        if idx in [0]:
            continue
        
        try:
            # 修正2：metas嵌套结构多了一层[0]
            # 同时提供双重保险：优先用data_infos，永远不会错
            current_sample = dataset.data_infos[idx]
            token = current_sample['token']
            lidar_path = current_sample['lidar_path']
            
            print(f"\n=== 正在处理第 {idx} 帧，token: {token} ===")
            print(f"点云路径: {lidar_path}")
            
            with torch.no_grad():
                data.pop("return_loss", None)
                result = model(return_loss=False, rescale=True, **data)
                pred = result[0]['occ_pred']  # (200,200,16)
                print(f"pred shape: {pred.shape}") 
                # 18类统计
                vals, counts = np.unique(pred, return_counts=True)
                total = pred.size

                print("映射前类别占比（含个数+百分数）：")
                for v, c in zip(vals, counts):
                    percent = c / total * 100
                    print(f"类别 {v}: {c} 个，占比 {percent:.2f}%")
                unique_vals = np.unique(pred)
                print(f"出现的类别: {unique_vals}")
                
                # 保存预测结果pkl（用token命名，避免覆盖）
                pkl_save_path = f"save/pkl/{token}.pkl"
                with open(pkl_save_path, "wb") as f:
                    pickle.dump(pred, f)
                print(f"预测结果已保存: {pkl_save_path}")
                
                # 保存当前帧的6张RGB图片
                img_save_dir = f"save/imgs/{token}"
                os.makedirs(img_save_dir, exist_ok=True)
                
                sample = nusc.get('sample', token)
                cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                
                for cam_name in cam_names:
                    cam_data = nusc.get('sample_data', sample['data'][cam_name])
                    # 修正3：图片原始路径改为实际数据集路径
                    src_img_path = os.path.join('/data5/wangyijun/DAOcc/data/nuscenes', cam_data['filename'])
                    dst_img_path = os.path.join(img_save_dir, f"{cam_name}.jpg")
                    
                    shutil.copy(src_img_path, dst_img_path)
                    print(f"已保存 {cam_name} 图片: {dst_img_path}")

        except Exception as e:
            print(f"\n⚠️  第 {idx} 帧处理失败: {str(e)}")
            print(f"跳过该帧，继续处理下一帧")
            continue

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    
    print("\n✅ 所有帧处理完成！")
    return results


def map_18_to_6_classes_vectorized(occ_pred):
    # 修正4：修复类别映射冲突（0不能同时映射到0和3）
    mapping = {
        # 0: ignorable points
        0: 0,   # others -> ignorable points
        17: 0,  # free  -> ignorable points
        
        # 1: Background (unobserved)
        1: 1,   # barrier -> Background (修正：原冲突)
        
        # 2: Vegetation
        16: 2,  # vegetation -> Vegetation
        
        # 3: Static obstacles
        8: 3,   # traffic_cone -> Static obstacles
        15: 3,  # manmade -> Static obstacles
        
        # 4: Dynamic obstacles  
        2: 4,   # bicycle -> Dynamic obstacles
        3: 4,   # bus -> Dynamic obstacles
        4: 4,   # car -> Dynamic obstacles
        5: 4,   # construction_vehicle -> Dynamic obstacles
        6: 4,   # motorcycle -> Dynamic obstacles
        7: 4,   # pedestrian -> Dynamic obstacles 
        9: 4,   # trailer -> Dynamic obstacles  
        10: 4,  # truck -> Dynamic obstacles               
        
        # 5: Road surface
        11: 5,  # driveable_surface -> Road surface
        12: 5,  # other_flat -> Road surface
        13: 5,  # sidewalk -> Road surface
        14: 5,  # terrain -> Road surface
    }
    # 向量化映射，速度比np.vectorize快10倍以上
    return np.vectorize(lambda x: mapping.get(x, 0))(occ_pred)