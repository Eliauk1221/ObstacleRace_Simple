import joblib
import torch
import os

start_idx = 0
end_idx = -1

def process_rawframes(data, dt, start_idx, end_idx):
    # root_pos = torch.tensor(data['root_trans_offset'], dtype=torch.double, device='cpu', requires_grad=False)[start_idx:end_idx]
    root_pos = torch.tensor(data['root_pos'], dtype=torch.double, device='cpu', requires_grad=False)[start_idx:end_idx]
    root_rot = torch.tensor(data['root_rot'], dtype=torch.double, device='cpu', requires_grad=False)[start_idx:end_idx]
    dof_pos = torch.tensor(data['dof_pos'], dtype=torch.double, device='cpu', requires_grad=False)[start_idx:end_idx]   
    # dof_pos[:, [5, 11]] = 0.0
    # root_pos[:, 2] += 0.1
    # dof_pos[:, 15] += 0.2
    # dof_pos[:, 20] += -0.2

    frames_tensor = torch.cat((root_pos[1:], root_rot[1:], dof_pos[1:]), dim=1)

    
    return frames_tensor.tolist()

if __name__ == "__main__":
    from pathlib import Path
    MOTION_FILE_Dir = "/home/sch/Documents/motion/e1/23dof/pingpang/pkl"
    OUTPUT_FILE_Dir = "/home/sch/Documents/motion/e1/23dof/pingpang/csv"
    folder = Path(MOTION_FILE_Dir)
    
    # 查找所有.pkl文件
    pkl_files = folder.rglob("*.pkl")
    for pkl_file in pkl_files:
        # csv_file = Path(str(pkl_file).replace('/pkl/', '/csv/').replace('.pkl', '.csv'))
        csv_file = Path(str(pkl_file).replace(MOTION_FILE_Dir, OUTPUT_FILE_Dir).replace('.pkl', '.csv'))
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        with open(pkl_file, 'rb') as file:
            motion = joblib.load(file)

        print(motion.keys())
        # for k, data in motion.items():

        #     frames_dt = 1. / data["fps"]
        #     frames = data['dof'].shape[0]
        #     print("Total motion frames: %d, Total motion durition: %f" %(frames, frames * frames_dt))

        #     # 写入CSV文件
        #     # 获取处理后的帧数据
        #     processed_data = process_rawframes(data, frames_dt, start_idx, end_idx)
        #     import csv
        #     # 保存为CSV格式
        #     with open(csv_file, "w") as f:
        #         # 检查processed_data的数据结构
        #         if isinstance(processed_data, list) and len(processed_data) > 0:
        #             # 如果是字典列表，提取键作为列标题
        #             if isinstance(processed_data[0], dict):
        #                 # 获取所有键作为列标题
        #                 fieldnames = processed_data[0].keys()
        #                 writer = csv.DictWriter(f, fieldnames=fieldnames)
        #                 writer.writeheader()
        #                 writer.writerows(processed_data)
        #             # 如果是列表的列表
        #             elif isinstance(processed_data[0], list):
        #                 writer = csv.writer(f)
        #                 writer.writerows(processed_data)
        #     print("Generating " + str(csv_file))
        
        
        ####### single motion #########
        data = motion
        frames_dt = 1. / data["fps"]
        frames = data['dof_pos'].shape[0]
        print("Total motion frames: %d, Total motion durition: %f" %(frames, frames * frames_dt))

        # 写入CSV文件
        # 获取处理后的帧数据
        processed_data = process_rawframes(data, frames_dt, start_idx, end_idx)
        import csv
        # 保存为CSV格式
        with open(csv_file, "w") as f:
            # 检查processed_data的数据结构
            if isinstance(processed_data, list) and len(processed_data) > 0:
                # 如果是字典列表，提取键作为列标题
                if isinstance(processed_data[0], dict):
                    # 获取所有键作为列标题
                    fieldnames = processed_data[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(processed_data)
                # 如果是列表的列表
                elif isinstance(processed_data[0], list):
                    writer = csv.writer(f)
                    writer.writerows(processed_data)
        print("Generating " + str(csv_file))