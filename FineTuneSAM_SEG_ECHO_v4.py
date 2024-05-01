import torch
import os
import wandb
import json
import evaluate
import multiprocessing
import numpy as np
import random
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments
from transformers import Trainer
from torchvision.transforms import ColorJitter
from transformers import (
    SegformerImageProcessor,
)
from torch import nn
# import NewProcessor
# start a new wandb run to track this script
# wandb.init()

# List all available GPUs
if torch.cuda.is_available():
    print("Available CUDA Devices:")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices are available")
# Check CUDA_VISIBLE_DEVICES environment variable
cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
print("CUDA_VISIBLE_DEVICES:", cuda_visible_devices if cuda_visible_devices is not None else "Not set (All GPUs are visible)")

# from huggingface_hub import notebook_login
# # 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
# notebook_login()
hf_dataset_identifier = "unreal-hug/REAL_DATASET_SEG_331"
# semantic_dataset.push_to_hub(hf_dataset_identifier)
filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
print("Id2label:", id2label)

processor = SegformerImageProcessor()
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)



def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs
# def train_transforms_v2(example_batch):
#     images = []; labels = []
#     for image, label in zip(example_batch['pixel_values'], example_batch['label']):
#         new_image, new_label = NewProcessor(image, label)
#         images.append(new_image)
#         labels.append(new_label)
#     inputs = processor(images, labels)
#     return inputs
def val_transforms(example_batch):
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            # scale the logits to the size of the label
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = metric._compute(
                    predictions=pred_labels,
                    references=labels,
                    num_labels=len(id2label),
                    ignore_index=0,
                    reduce_labels=processor.do_reduce_labels,
                )

            # add per category metrics as individual key-value pairs
            per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
            per_category_iou = metrics.pop("per_category_iou").tolist()

            metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
            metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

            return metrics

# pretrained_models = ["nvidia/mit-b0","nvidia/mit-b1","nvidia/mit-b2","nvidia/mit-b3"]
pretrained_models = ["nvidia/mit-b3", "nvidia/mit-b4"]
hub_model_id = "250_experiments_model_new_v1"
seeds_f_name = "MyTrainingSeedsV4.txt"

# rand_seed = np.random.choice(range(100, 999), 2, replace=False)
# with open(seeds_f_name, 'a') as f:
#     np.savetxt(f, rand_seed, delimiter=',', fmt='%d')

# Open the file in read mode ('r') and use readlines() to read all lines into a list
with open('MyTrainingSeedsV4.txt', 'r') as file:
    lines = file.readlines()
# Optionally, you might want to strip the newline characters from each line
rand_seed = [int(line.rstrip('\n')) for line in lines]
size = len(rand_seed)
# print(rand_seed)
for pretrained_model_name in pretrained_models:
    cnt = 0
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # Move your model to the selected device
    metric = evaluate.load("mean_iou")
    # metric = evaluate.load("accuracy")    
    # 25,25,25,25,50,50,50,50, 
    # 0.001,0.0001,0.00001,0.000001,0.001,0.0001,0.00001,0.000001
    # 8,8,8,8,16,16,16,16,

    while cnt < size:
        
        the_seed = rand_seed[cnt]
        # if pretrained_model_name == "nvidia/mit-b4" and the_seed > 100:
        #     continue
        project_name = "-".join([pretrained_model_name.split(sep="-")[-1],"250-8-1e5"])#"250-16-1e4"

        ds = load_dataset(hf_dataset_identifier, split='all')
        ds = ds.shuffle(seed=the_seed)
        ds = ds.train_test_split(test_size=0.2)
        train_ds = ds["train"]
        print(train_ds)
        test_ds = ds["test"]
        print(test_ds)
        # Set transforms
        train_ds.set_transform(train_transforms)
        test_ds.set_transform(val_transforms)

        wandb.init(
            project=project_name,
            name="-".join(["rand-seed",str(the_seed)]),
            config={"epochs": 250, "batch_size": 8, "learning_rate":1e-5},
            # config={"epochs": 50, "batch_size": 8, "learning_rate":1e-5},
            # config={"epochs": 25, "batch_size": 8, "learning_rate":1e-5},
        )
        
        training_args = TrainingArguments(
        output_dir="segments-ECHO-331-split-all",
        learning_rate=wandb.config.learning_rate,
        num_train_epochs=wandb.config.epochs,
        per_device_train_batch_size=wandb.config.batch_size,
        per_device_eval_batch_size=wandb.config.batch_size,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        eval_steps=200,
        logging_steps=10,
        eval_accumulation_steps=5,
        load_best_model_at_end=True,
        hub_strategy="end",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics,
            )
        # with torch.no_grad():
        trainer.train()
        wandb.finish()
        torch.cuda.empty_cache()
        print(f"Model: {pretrained_model_name}")
        print(f"STEP: {cnt+1} DONE!!!")
        cnt += 1