import argparse
from time import time
from tqdm import tqdm
from data import prepare_dataloaders_ptm
from model import prepare_models_secondary_structure_ptm
import pandas as pd
import yaml
import torch.nn.functional as F
import os
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
from box import Box

# device: CPU or GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def remove_label(tensor, output, label):
    mask = tensor != label
    return tensor[mask], output[mask]


def plot_prompt(prompt_layer_dict, epoch, result_path):

    # 1) collect the data
    data = np.concatenate([
        tensor.detach().cpu().numpy()
        for tensor in prompt_layer_dict.values()
    ], axis=0)

    # 2) numeric labels
    labels = np.concatenate([
        np.full(prompt_layer_dict[k].shape[0], int(k), dtype=int)
        for k in prompt_layer_dict.keys()
    ])

    # 3) UMAP
    umap_model = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    umap_result = umap_model.fit_transform(data)

    # 4) Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap_result[:, 0],
        umap_result[:, 1],
        c=labels,
        cmap='viridis',
        s=10
    )
    plt.title('UMAP Projection of Prompt Embeddings')
    plt.colorbar(scatter, label='Task ID')

    # 5) ensure output dir exists
    out_dir = os.path.join(result_path, 'prompt_figure', f'epoch {str(epoch)}')
    os.makedirs(out_dir, exist_ok=True)

    # 6) save
    plt.savefig(os.path.join(out_dir, 'prompt_dict_comparison.png'), dpi=300)
    print(f'The prompt figure of epoch {epoch} is saved!')
    plt.close()


def predict(dataloader, net):
    counter = 0
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description("Steps")

    prediction_results = []
    prot_id_results = []
    position_results = []

    for i, data in enumerate(dataloader):
        prot_id, sequences, masks, task_ids, indices = data

        # ---- move everything needed to the same device ----
        sequences = sequences.to(device)
        masks = masks.to(device)
        task_ids = task_ids.to(device)

        with torch.inference_mode():
            # x_contact, batch, length, out_channels
            outputs = net(sequences, task_ids)

            # boolean / 0-1 mask is now on the same device as outputs
            preds = outputs[masks]
            if preds.numel() == 0:
                continue

            # compute positions from mask
            pos_idx = (masks[0] == 1).nonzero(as_tuple=True)[0].tolist()
            positions = [v + 1 for v in pos_idx]

            preds = F.softmax(preds, dim=-1)[:, 1]
            rounded_preds = [round(v, 3) for v in preds.tolist()]
            prediction_results.extend(rounded_preds)
            position_results.extend(positions)

            for _ in range(len(positions)):
                prot_id_results.append(prot_id[0])

        counter += 1
        progress_bar.update(1)

    return prediction_results, prot_id_results, position_results


def main(args, dict_config):
    configs = Box(dict_config)

    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()

    dataloaders_dict = prepare_dataloaders_ptm(args, configs)
    net = prepare_models_secondary_structure_ptm(configs)

    # ---- move model to the same device as we use for data ----
    net.to(device)

    # load checkpoint on the same device
    model_checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    net.load_state_dict(model_checkpoint['model_state_dict'])

    for i, (task_name, dataloader) in enumerate(dataloaders_dict['test'].items()):
        net.eval()
        start_time = time()
        prediction_results, prot_id_results, position_results = predict(dataloader, net)
        end_time = time()

        # save prediction results into csv
        result_dic = {
            "prot_id": prot_id_results,
            "position": position_results,
            "prediction": prediction_results,
        }

        df = pd.DataFrame(result_dic)
        save_path = os.path.join(args.save_path, task_name + '_test_output.csv')
        df.to_csv(save_path, index=False)
        print("The prediction has been saved in " + save_path + ".")

        print("prediction time:", end_time - start_time)

    del net, dataloaders_dict
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--config_filename", "-c", default='config/test.yaml')
    parser.add_argument("--model_path", default='checkpoints/best_model_yichuan.pth')
    parser.add_argument("--data_path", default='data/Phosphorylation_ST_sequence.fasta')
    parser.add_argument("--save_path", default='data')
    parser.add_argument("--PTM_type", default='Phosphorylation_ST')

    args = parser.parse_args()

    config_filename = args.config_filename
    with open(config_filename) as file:
        config_file = yaml.full_load(file)

    main(args, config_file)
