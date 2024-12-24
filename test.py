import os
import time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import build_unet
from utils import create_dir, seeding


def calculate_metrics(y_true, y_pred):
    """ Calculate evaluation metrics """
    y_true = y_true.cpu().numpy().astype(np.uint8).reshape(-1)
    y_pred = (y_pred.cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def mask_parse(mask):
    """ Convert grayscale mask to RGB """
    mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (H, W, 3)
    return mask


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    results_dir = os.path.join("results")
    create_dir(results_dir)

    """ Load dataset """
    test_x = sorted(glob(os.path.join("data", "test", "image", "*")))
    test_y = sorted(glob(os.path.join("data", "test", "mask", "*")))

    assert len(test_x) > 0, "No test images found. Check 'data/test/image/' directory."
    assert len(test_y) > 0, "No test masks found. Check 'data/test/mask/' directory."

    """ Hyperparameters """
    H, W = 512, 512
    size = (W, H)
    checkpoint_path = os.path.join("files", "checkpoint.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    """ Load the checkpoint """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        try:
            """ Extract the name """
            name = os.path.splitext(os.path.basename(x))[0]

            """ Reading image """
            image = cv2.imread(x, cv2.IMREAD_COLOR)  # (H, W, 3)
            if image is None:
                print(f"Failed to read image: {x}")
                continue
            image = cv2.resize(image, size)
            x_input = np.transpose(image, (2, 0, 1))  # (3, H, W)
            x_input = x_input / 255.0
            x_input = np.expand_dims(x_input, axis=0).astype(np.float32)  # (1, 3, H, W)
            x_input = torch.from_numpy(x_input).to(device)

            """ Reading mask """
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  # (H, W)
            if mask is None:
                print(f"Failed to read mask: {y}")
                continue
            mask = cv2.resize(mask, size)
            mask_np = mask / 255.0  # Normalize mask for visualization

            y_target = np.expand_dims(mask, axis=0)  # (1, H, W)
            y_target = np.expand_dims(y_target, axis=0) / 255.0  # (1, 1, H, W)
            y_target = torch.from_numpy(y_target.astype(np.float32)).to(device)

            with torch.no_grad():
                """ Prediction and calculating FPS """
                start_time = time.time()
                pred_y = model(x_input)
                pred_y = torch.sigmoid(pred_y)
                total_time = time.time() - start_time
                time_taken.append(total_time)

                """ Calculate metrics """
                score = calculate_metrics(y_target, pred_y)
                metrics_score = list(map(add, metrics_score, score))

                """ Post-process prediction """
                pred_y = pred_y[0].cpu().numpy().squeeze()  # (H, W)
                pred_y = (pred_y > 0.5).astype(np.uint8)  # Binary mask

                # Debug: Check unique values in prediction and mask
                print(f"Prediction unique values: {np.unique(pred_y)}")
                print(f"Mask unique values: {np.unique(mask_np)}")

            """ Saving masks """
            ori_mask = mask_parse(mask_np * 255)  # Convert normalized mask to RGB
            pred_mask = mask_parse(pred_y * 255)  # Convert prediction to RGB
            line = np.ones((H, 10, 3)) * 128  # Separator line

            combined_image = np.concatenate(
                [image, line, ori_mask, line, pred_mask], axis=1
            )  # Concatenate input, mask, and prediction
            save_path = os.path.join(results_dir, f"{name}.png")
            if cv2.imwrite(save_path, combined_image):
                print(f"Saved result: {save_path}")
            else:
                print(f"Failed to save result: {save_path}")

        except Exception as e:
            print(f"Error processing {x}: {e}")

    """ Final metrics """
    jaccard = metrics_score[0] / len(test_x)
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    print(f"Jaccard: {jaccard:.4f} - F1: {f1:.4f} - Recall: {recall:.4f} - Precision: {precision:.4f} - Acc: {acc:.4f}")

    fps = 1 / np.mean(time_taken)
    print("FPS: ", fps)
