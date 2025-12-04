import torch
import yaml
import numpy as np
import pandas as pd
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from Bio import SeqIO

from transformers import AutoTokenizer, T5Tokenizer


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def find_indexes(string, chars):
    indexes = []
    for i, char in enumerate(string):
        # Check if the character is in the chars list and not in the exclude list
        if char in chars:
            # Convert the 0-indexed position back to 1-indexed for the result
            indexes.append(i + 1)
    return indexes

def check_ptm_site(sequence, positions, allowed_ptm_sites):
    for i in positions.copy():
        if i > len(sequence):
            positions.remove(i)
            continue
        elif not sequence[i - 1] in allowed_ptm_sites:
            positions.remove(i)
    return positions

def check_center_amino_acid(sequence, position, positive_amino_acids):
   if sequence[position - 1] in positive_amino_acids:
       return True

def extract_positions(sequence,configs):
    ptm_position= {}
    token_info = configs.encoder.condition_token.token_info
    for i in range(len(sequence)):
        if sequence[i] in token_info.keys():
            ptm_position[i]=token_info[sequence[i]]
        else:
            ptm_position[i]="None"

    return ptm_position



class PTMDataset(Dataset):
    def __init__(self, ids,sequences,masks,task_ids, configs):
        # self.samples_list=samples_list
        self.ids = ids
        self.sequences = sequences #all the sequences

        self.masks = masks

        self.task_ids=task_ids
        self.configs = configs

        if  self.configs.encoder.model_name.startswith("esmc"):
            self.max_length = configs.encoder.max_len
            # self.client = ESMC.from_pretrained("esmc_600m").to(device)  # or "cpu"
        else:
            # self.esm2_model, self.esm2_alphabet = prepare_adapter_h_model(configs_all, logging)
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)
            self.max_length = configs.encoder.max_len


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        prot_id, sequence, mask, task_token, index = (
            self.ids[index], self.sequences[index],self.masks[index], self.task_ids[index],index)

        if  self.configs.encoder.model_name.startswith("esmc"):
            encoded_sequence = sequence

            padded_mask = np.array(mask)
        else:
            encoded_sequence = self.encoder_tokenizer(sequence, max_length=self.max_length + 2, padding='max_length',
                                                      truncation=True,
                                                      return_tensors="pt"
                                                      )
            encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
            encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])

            padded_mask = np.pad(mask, (0, self.max_length - len(sequence)), 'constant')
            padded_task_token = np.pad(task_token, (0, self.max_length - len(sequence)), 'constant')


        # return prot_id, encoded_sequence, padded_label, padded_mask.astype(bool), task_token, padded_positive_masks.astype(bool),padded_negative_masks.astype(bool),index
        return prot_id, encoded_sequence,  padded_mask.astype(bool), padded_task_token,index


def split_into_chunks(id, sequence, label, mask,task_token, max_length, overlap):
    chunked_ids = []
    chunked_sequences = []
    chunked_labels = []
    chunked_masks = []
    chunked_mask_positive=[]
    chunked_mask_negative=[]
    chunked_task_ids=[]

    start = 0
    p = 1
    while start < len(sequence):
        end = min(start + max_length, len(sequence))
        if 1 in mask[start:end]:
            chunked_sequences.append(sequence[start:end])
            chunked_labels.append(label[start:end])
            chunked_masks.append(mask[start:end])
            chunked_ids.append(id + "_" + str(p))
            chunked_task_ids.append(task_token)
            p += 1

        start += max_length - overlap


    return chunked_ids, chunked_sequences, chunked_labels, chunked_masks,chunked_task_ids


def split_into_chunks_with_ptm_resolution(id, sequence, label, mask,task_token,
                                          max_length, overlap):
    """
    Enhanced version of your chunking function with PTM conflict resolution.

    Parameters:
    - id: sequence identifier
    - sequence: protein sequence
    - label: labels for each position
    - mask: general mask (1 for positions of interest)
    - mask_positive: positive mask
    - mask_negative: negative mask
    - task_token: task identifier
    - max_length: maximum chunk length
    - overlap: overlap between chunks

    Returns: Same as original but with PTM conflict resolution
    """
    # First, collect all potential chunks
    potential_chunks = []
    start = 0
    p = 1

    while start < len(sequence):
        end = min(start + max_length, len(sequence))
        chunk_mask = mask[start:end]

        if 1 in chunk_mask:  # Only keep chunks with at least one masked position
            chunk_data = {
                'id': id + "_" + str(p),
                'sequence': sequence[start:end],
                'label': label[start:end],
                'mask': chunk_mask,
                'task_token': task_token[start:end],
                'start_pos': start,
                'end_pos': end,
                'chunk_num': p
            }
            potential_chunks.append(chunk_data)
            p += 1

        start += max_length - overlap

    # Resolve PTM conflicts (positions where mask=1 appears in multiple chunks)
    final_chunks = resolve_ptm_conflicts(potential_chunks, mask)

    # Extract results in original format
    chunked_ids = []
    chunked_sequences = []
    chunked_labels = []
    chunked_masks = []

    chunked_task_ids = []

    for chunk in final_chunks:
        chunked_ids.append(chunk['id'])
        chunked_sequences.append(chunk['sequence'])
        chunked_labels.append(chunk['label'])
        chunked_masks.append(chunk['mask'])
        chunked_task_ids.append(chunk['task_token'])

    return chunked_ids, chunked_sequences, chunked_labels, chunked_masks,chunked_task_ids


def resolve_ptm_conflicts(chunks, original_mask, strategy='most_central'):
    """
    Resolve conflicts when the same PTM position appears in multiple chunks.

    Parameters:
    - chunks: list of chunk dictionaries
    - original_mask: original mask to identify PTM positions
    - strategy: 'most_central', 'first_occurrence', 'longest_chunk', or 'fewest_ptms'
    """
    if not chunks:
        return chunks

    # Find all PTM positions (where original_mask = 1)
    ptm_positions = [i for i, val in enumerate(original_mask) if val == 1]

    # Track which chunks contain each PTM
    ptm_to_chunks = {}
    for ptm_pos in ptm_positions:
        ptm_to_chunks[ptm_pos] = []
        for chunk in chunks:
            if chunk['start_pos'] <= ptm_pos < chunk['end_pos']:
                ptm_to_chunks[ptm_pos].append(chunk)

    # Resolve conflicts based on strategy
    ptm_assignments = {}  # ptm_position -> chunk_id

    for ptm_pos, conflicting_chunks in ptm_to_chunks.items():
        if len(conflicting_chunks) <= 1:
            # No conflict
            if conflicting_chunks:
                ptm_assignments[ptm_pos] = conflicting_chunks[0]['id']
        else:
            # Resolve conflict based on strategy
            if strategy == 'most_central':
                # Keep PTM in chunk where it's most central
                best_chunk = min(conflicting_chunks,
                                 key=lambda c: abs((c['start_pos'] + c['end_pos']) / 2 - ptm_pos))
            elif strategy == 'first_occurrence':
                # Keep in first chunk
                best_chunk = min(conflicting_chunks, key=lambda c: c['chunk_num'])
            elif strategy == 'longest_chunk':
                # Keep in longest chunk
                best_chunk = max(conflicting_chunks,
                                 key=lambda c: len(c['sequence']))
            elif strategy == 'fewest_ptms':
                # Keep in chunk with fewest other PTMs
                chunk_ptm_counts = {}
                for chunk in conflicting_chunks:
                    count = sum(1 for pos in ptm_positions
                                if chunk['start_pos'] <= pos < chunk['end_pos'])
                    chunk_ptm_counts[chunk['id']] = count
                best_chunk = min(conflicting_chunks,
                                 key=lambda c: chunk_ptm_counts[c['id']])
            else:
                # Default to most central
                best_chunk = min(conflicting_chunks,
                                 key=lambda c: abs((c['start_pos'] + c['end_pos']) / 2 - ptm_pos))

            ptm_assignments[ptm_pos] = best_chunk['id']

    # Update chunk masks to remove PTMs assigned to other chunks
    final_chunks = []
    for chunk in chunks:
        # Create new masks with only assigned PTMs
        new_mask = [0] * len(chunk['mask'])

        new_label = chunk['label'][:]  # Copy original labels

        has_ptm = False
        for i, (mask_val, orig_label) in enumerate(zip(chunk['mask'], chunk['label'])):
            global_pos = chunk['start_pos'] + i

            if global_pos in ptm_assignments and ptm_assignments[global_pos] == chunk['id']:
                # This PTM is assigned to this chunk
                new_mask[i] = mask_val

                has_ptm = True
            elif mask_val == 1:
                # This PTM is assigned to another chunk, zero out the masks
                new_mask[i] = 0

                # You might want to adjust the label as well
                # new_label[i] = 0  # Uncomment if you want to zero out labels too

        # Only keep chunks that still have at least one PTM after conflict resolution
        if has_ptm:
            chunk_copy = chunk.copy()
            chunk_copy['mask'] = new_mask
            chunk_copy['label'] = new_label
            final_chunks.append(chunk_copy)

    return final_chunks

def prepare_task(dataset_path, task_token, positive_amino_acids, max_length,chunk):

    ids = []
    sequences = []
    masks = []
    task_ids=[]

    fasta_file = dataset_path
    data_list = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        data_dic = {}
        data_dic['id'] = record.id
        data_dic['sequence'] = str(record.seq)
        data_list.append(data_dic)
    df = pd.DataFrame(data_list)

    for row in df.itertuples():

        sequence = row.sequence
        prot_id = row.id

        valid_mask = [0] * len(sequence)

        all_positions = find_indexes(sequence, positive_amino_acids)
        for position in all_positions:
            valid_mask[position - 1] = 1

        task_token_list=valid_mask.copy()

        for i in range(len(sequence)):
            task_token_list[i]=task_token

        if chunk:
            chunked_ids, chunked_sequences, chunked_masks,chunked_task_ids = split_into_chunks_with_ptm_resolution(prot_id, sequence, valid_mask, task_token_list, max_length, 100)

            ids.extend(chunked_ids)
            sequences.extend(chunked_sequences)
            labels.extend(chunked_labels)
            masks.extend(chunked_masks)
            task_ids.extend(chunked_task_ids)
            # samples.append((prot_id,sequence,label,teacher_distill_output[prot_id],valid_mask,task_token,valid_mask_positive,valid_mask_negative))
        else:
            ids.append(prot_id)
            sequences.append(sequence)
            masks.append(valid_mask)
            task_ids.append(task_token)
            #
            #
            #
            # if type=="train":
            #     samples.append((prot_id,sequence,label,teacher_distill_output[prot_id],valid_mask,task_token,valid_mask_positive,valid_mask_negative))
            # else:
            #     samples.append((prot_id, sequence, label, "none", valid_mask, task_token,
            #                     valid_mask_positive, valid_mask_negative))
            # else:
            #     samples.append((prot_id, sequence, label, valid_mask, task_token,valid_mask_positive,valid_mask_negative))

    return ids,sequences,masks,task_ids




def get_test_samples(configs, args,task_info):
    ids,sequences,masks,task_ids = prepare_task(
        dataset_path= args.data_path,
        task_token=task_info['id'],
        positive_amino_acids=task_info['ptm_amino_acid'],
        max_length=configs.encoder.max_len,
        chunk=configs.test_settings.chunk
    )


    dataset_test = PTMDataset(ids,sequences,masks,task_ids, configs)
    test_loader = DataLoader(dataset_test, batch_size=configs.test_settings.batch_size,
                             shuffle=True, pin_memory=False, drop_last=False,
                             num_workers=configs.test_settings.num_workers)

    return test_loader

def prepare_dataloaders_ptm(args,configs):
    task_list=[]
    dataloaders_dict_test= {}

    if args.PTM_type=='Phosphorylation_ST':
        task_list.append(
            {'task_name':"Phosphorylation_ST",'id':configs.task_ids.Phosphorylation_ST,'file_name':args.data_path,
                          'ptm_amino_acid':["S","T"]})

    if args.PTM_type=="Phosphorylation_Y":
        task_list.append(
            {'task_name': "Phosphorylation_Y", 'id': configs.task_ids.Phosphorylation_Y, 'file_name': args.data_path,
             'ptm_amino_acid': ["Y"]})

    if args.PTM_type=="Ubiquitination_K":
        task_list.append(
            {'task_name': "Ubiquitination_K", 'id': configs.task_ids.Ubiquitination_K, 'file_name': args.data_path,
             'ptm_amino_acid': ["K"]})

    if args.PTM_type=="Acetylation_K":
        task_list.append(
            {'task_name': "Acetylation_K", 'id': configs.task_ids.Acetylation_K, 'file_name': args.data_path,
             'ptm_amino_acid': ["K"]})

    if args.PTM_type=="Methylation_R":
        task_list.append(
            {'task_name': "Methylation_R", 'id': configs.task_ids.Methylation_R, 'file_name': args.data_path,
             'ptm_amino_acid': ["R"]})

    if args.PTM_type=="NlinkedGlycosylation_N":
        task_list.append(
            {'task_name': "NlinkedGlycosylation_N", 'id': configs.task_ids.NlinkedGlycosylation_N, 'file_name': args.data_path,
             'ptm_amino_acid': ["N"]})

    if args.PTM_type=="Methylation_K":
        task_list.append(
            {'task_name': "Methylation_K", 'id': configs.task_ids.Methylation_K, 'file_name': args.data_path,
             'ptm_amino_acid': ["K"]})

    if args.PTM_type=="Sumoylation_K":
        task_list.append(
            {'task_name': "Sumoylation_K", 'id': configs.task_ids.Sumoylation_K, 'file_name': args.data_path,
             'ptm_amino_acid': ["K"]})


    for task_info in task_list:
        dataloaders_dict_test[task_info['task_name']] = get_test_samples(configs, args, task_info)

    return {'test': dataloaders_dict_test}

