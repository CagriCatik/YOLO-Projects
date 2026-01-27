import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import os

def csv_to_tensorboard():
    csv_path = "runs/detect/traffic_sign_detection/yolo11_custom/results.csv"
    log_dir = "runs/detect/traffic_sign_detection/yolo11_custom"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Converting {csv_path} to TensorBoard logs...")
    df = pd.read_csv(csv_path)
    
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    
    writer = SummaryWriter(log_dir=log_dir)
    
    for _, row in df.iterrows():
        epoch = int(row['epoch'])
        for col in df.columns:
            if col != 'epoch':
                writer.add_scalar(col, row[col], epoch)
                
    writer.close()
    print(f"Conversion complete! Logs saved to {log_dir}")
    print("You can now run: python -m tensorboard.main --logdir runs/")

if __name__ == "__main__":
    csv_to_tensorboard()
