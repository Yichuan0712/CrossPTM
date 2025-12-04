import torch
import torch.nn as nn
import esm_adapterH
import esm
import numpy as np
from peft import LoraConfig, get_peft_model
from transformers.tokenization_utils_base import BatchEncoding
from esm_adapterH.prompt_tuning import PrefixTuning

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from model_coattention_tranception import CoattentiontraceptionBlock
from linearmix import MLPMixPrompt

def verify_data_types(model, logging=None):
    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        if logging:
            logging.info(f"{k}, {v}, {v / total}")


def prepare_adapter_h_model(configs, logging=None):
    if logging:
        logging.info("use adapterH ESM model")

    adapter_args = configs.encoder.adapter_h
    model_name = configs.encoder.model_name.split('/')[-1]

    # Create the model dynamically using module attributes
    model_constructor = getattr(esm_adapterH.pretrained, model_name, None)
    model, alphabet = model_constructor(adapter_args)
    num_layers = model.num_layers
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    if configs.encoder.prompt.enable:
        if not hasattr(configs.encoder.prompt, "num_tasks"):
            configs.encoder.prompt.num_tasks = 1
        model.prefix_module = PrefixTuning(configs, model,
                                           prompt_len=configs.encoder.prompt.prompt_len,
                                           prompt_layer_indices=configs.encoder.prompt.prompt_layer_indices,
                                           num_tasks=configs.encoder.prompt.num_tasks
                                           )
        if configs.encoder.prompt.if_grads:
            for param in model.prefix_module.parameters():
                param.requires_grad = True
        else:
            for param in model.prefix_module.parameters():
                param.requires_grad = False

    if configs.encoder.adapter_h.enable:
        if not isinstance(configs.encoder.adapter_h.freeze_adapter_layers, list):
            configs.encoder.adapter_h.freeze_adapter_layers = [configs.encoder.adapter_h.freeze_adapter_layers]

    if configs.encoder.fine_tune.enable:
        if not isinstance(configs.encoder.fine_tune.freeze_adapter_layers, list):
            configs.encoder.fine_tune.freeze_adapter_layers = [configs.encoder.fine_tune.freeze_adapter_layers]

    if configs.encoder.lora.enable:
        if logging:
            logging.info('enable LoRa on top of adapterH model')
        if hasattr(configs.encoder.lora, "lora_targets"):
            lora_targets = configs.encoder.lora.lora_targets
        else:
            lora_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                            "self_attn.out_proj"]
        target_modules = []
        if configs.encoder.lora.esm_num_end_lora > 0:
            start_layer_idx = np.max([num_layers - configs.encoder.lora.esm_num_end_lora, 0])
            for idx in range(start_layer_idx, num_layers):
                for layer_name in lora_targets:
                    target_modules.append(f"layers.{idx}.{layer_name}")

        config = LoraConfig(
            r=configs.encoder.lora.r,
            lora_alpha=configs.encoder.lora.lora_alpha,
            target_modules=target_modules,
            inference_mode=False,
            lora_dropout=configs.encoder.lora.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, config)

        verify_data_types(model, logging)

    elif not configs.encoder.lora.enable and configs.encoder.fine_tune.enable:
        # fine-tune the latest layer

        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.layers[-configs.encoder.fine_tune.last_layers_trainable:].parameters():
            param.requires_grad = True

        # if you need fine-tune last layer, the emb_layer_norm_after for last representation should be updated
        if configs.encoder.fine_tune.last_layers_trainable != 0:
            for param in model.emb_layer_norm_after.parameters():
                param.requires_grad = True

    # only freeze all the parameters once at the beginning. then open some layers later
    # only make adapterH trainable according to freeze_adapter_layers
    if configs.encoder.adapter_h.enable:
        for adapter_idx, value in enumerate(configs.encoder.adapter_h.freeze_adapter_layers):
            if not value:
                for name, param in model.named_parameters():
                    adapter_name = f"adapter_{adapter_idx}"
                    if adapter_name in name:
                        # Freeze all parameters by default
                        param.requires_grad = True

    # only freeze all the parameters once at the beginning. then open some layers later,but because
    # of fine_tune, adapter layers might be tunable.
    # change on 1/15/2024 not need to use freeze_adapter_layers to control fine-tune part! use another parameter instead and must after setting of freeze_adapter_layers
    if configs.encoder.fine_tune.enable:  # only see fine_tune.freeze_adapter_layers when fine-tune is available
        for adapter_idx, value in enumerate(configs.encoder.fine_tune.freeze_adapter_layers):
            if value:
                for name, param in model.named_parameters():
                    adapter_name = f"adapter_{adapter_idx}"
                    if adapter_name in name:
                        # Freeze all parameters by default
                        print("freeze adapter in fine-tune")
                        param.requires_grad = False
    # """

    if configs.encoder.tune_embedding:
        for param in model.embed_tokens.parameters():
            param.requires_grad = True

    # if configs.encoder.prompt.enable:
    #     for param in model.prefix_module.parameters():
    #         param.requires_grad = True
    if configs.encoder.prompt.enable:
        if configs.encoder.prompt.if_grads:
            for param in model.prefix_module.parameters():
                param.requires_grad = True
        else:
            for param in model.prefix_module.parameters():
                param.requires_grad = False

    return model, alphabet


def prepare_esm_model(configs, logging=None):
    if logging:
        logging.info("use ESM model")

    model_name = configs.encoder.model_name.split('/')[-1]

    # Create the model dynamically using module attributes
    model_constructor = getattr(esm.pretrained, model_name, None)
    model, alphabet = model_constructor()
    num_layers = model.num_layers
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    if configs.encoder.lora.enable:
        if logging:
            logging.info('enable LoRa on top of esm model')

        if hasattr(configs.encoder.lora, "lora_targets"):
            lora_targets = configs.encoder.lora.lora_targets
        else:
            lora_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                            "self_attn.out_proj"]
        target_modules = []
        if configs.encoder.lora.esm_num_end_lora > 0:
            start_layer_idx = np.max([num_layers - configs.encoder.lora.esm_num_end_lora, 0])
            for idx in range(start_layer_idx, num_layers):
                for layer_name in lora_targets:
                    target_modules.append(f"layers.{idx}.{layer_name}")

        config = LoraConfig(
            r=configs.encoder.lora.r,
            lora_alpha=configs.encoder.lora.lora_alpha,
            target_modules=target_modules,
            inference_mode=False,
            lora_dropout=configs.encoder.lora.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, config)

        verify_data_types(model, logging)

    elif not configs.encoder.lora.enable and configs.encoder.fine_tune.enable:
        # fine-tune the latest layer
        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.layers[-configs.encoder.fine_tune.last_layers_trainable:].parameters():
            param.requires_grad = True

        # if you need fine-tune last layer, the emb_layer_norm_after for last representation should be updated
        if configs.encoder.fine_tune.last_layers_trainable != 0:
            for param in model.emb_layer_norm_after.parameters():
                param.requires_grad = True
    elif hasattr(configs.encoder, "prompt"):

        if configs.encoder.prompt.enable:
            if not hasattr(configs.encoder.prompt, "num_tasks"):
                configs.encoder.prompt.num_tasks = 1

            model.prefix_module = PrefixTuning(model, prompt_len=configs.encoder.prompt.prompt_len,
                                               prompt_layer_indices=configs.encoder.prompt.prompt_layer_indices,
                                               )
            for param in model.prefix_module.parameters():
                param.requires_grad = True

    if configs.encoder.tune_embedding:
        if logging:
            logging.info('make esm embedding parameters trainable')

        for param in model.embed_tokens.parameters():
            param.requires_grad = True

    return model, alphabet


def retrieve_embedding(task_ids, embeddings):
    """
    Retrieve embeddings for the given task IDs.

    Args:
      task_ids (Tensor): A tensor containing the task IDs.
      embeddings (Tensor): A tensor containing the embeddings.

    Returns:
      Tensor: A tensor containing the retrieved embeddings.

    """
    # return embeddings[task_ids].squeeze(1)
    embedding_list = []
    for task_id in task_ids:
        embedding_list.append(embeddings[str(task_id[0].item())])
    return torch.stack(embedding_list)


def retrieve_inputs(task_ids, prompt_layer_dict):
    embedding_list = []
    for task_id in task_ids:
        embedding_list.append(prompt_layer_dict[str(task_id[0].item())])

    # Concatenate each key across the list
    merged = {
        key: torch.cat([enc[key] for enc in embedding_list], dim=0)
        for key in embedding_list[0].keys()
    }

    batch_inputs = BatchEncoding(merged)

    return batch_inputs


def from_sample_of_embeddings(embeddings, population_size=None):
    """Initialize by drawing vectors from the embedding table.

    Note:
      If not provided, the population size used is the full possibility of the
      vector space.

    Args:
      embeddings: [V, H] The embeddings to draw vectors from. can be extract
        by model_seq.esm2.embed_tokens.weight
      population_size: Limit the drawing to the first `population_size` vectors.

    Returns:
      A closure over the embedding table that can be used as a flax initializer.
    """
    if population_size is None:
        population_size = embeddings.shape[0]
    if population_size <= 0:
        raise ValueError(f"Cannot sample from a population less than zero. Got "
                         f"{population_size}")
    if population_size > embeddings.shape[0]:
        print("The requested `population_size` (%d) is larger than the "
              "total available embeddings (%d). Setting "
              "`population_size` to the embedding size.", population_size,
              embeddings.shape[0])

        population_size = embeddings.shape[0]

    # Because our sampling is done with jax (so that we can make use of the rng
    # key), we need our embeddings to be in jax, otherwise we get errors because
    # the indices will be a jax tracer and it fails when it is converted to numpy
    # to lookup values in a number array. This call pins the embeddings to cpu so
    # we don't waste TPU memory carrying it around.
    embeddings = embeddings.cpu()

    def initialize_from_embedding_sample(shape, rng):

        """Sample from the embedding table, without replacement.

        Note:
          If the number of prompt tokens requested is larger than the total number
          of vectors we are drawing from (`population_size`) we do sampling with
          replacement.

        Args:
          rng: The rng seed used in our sampling.
          shape: The shape of the prompt variable. shape[0] tells us how many
            vectors to sample.

        Raises:
          ValueError if the number of features in the embedding table do not match
          the number of features in the prompt.

        Returns:
          A sample of the embedding table as a jax array. [P, H]
        """
        if embeddings.shape[-1] != shape[-1]:
            raise ValueError(
                "Shape mismatch between the number of features in the "
                f"embeddings: {embeddings.shape[-1]} and the requested prompt shape "
                f"{shape[-1]}.")
        replace = False
        if shape[0] > population_size:
            # print("Prompt Length: %d is larger than the number of vectors "
            #       "to draw from: %d. Switching to draws with replacement.", shape[0],
            #       population_size)
            replace = True

        # set the seed for torch random number generator
        torch.manual_seed(rng)
        if replace:
            index = torch.randint(population_size, size=(shape[0],))
        else:
            index = torch.multinomial(torch.ones(
                population_size), shape[0], replacement=False)
        # print("task_id:" + str(rng) + ";index:" + str(index))

        return embeddings[index].clone().detach()

    return initialize_from_embedding_sample


class EncoderSSPTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.encoder.adapter_h.enable:
            self.esm2, self.alphabet = prepare_adapter_h_model(configs)
        else:
            self.esm2, self.alphabet = prepare_esm_model(configs)

        self.configs = configs

        if configs.crossattention:
            if configs.task == 'linear_mix':
                temperature = float(getattr(configs, "linear_mix_temperature", 1.0))
                init_std = float(getattr(configs, "linear_mix_init_std", 0.02))
                low_rank = getattr(configs, "linear_mix_low_rank", 64)  # None/整数
                use_low_rank = (low_rank is not None)

                num_tasks = getattr(configs.encoder.prompt, "num_tasks", getattr(configs, "task_number", 1))
                prompt_len = int(getattr(configs, "task_shape",getattr(configs.encoder.prompt, "prompt_len", 16)))
                embed_size = int(getattr(configs, "task_dimension", 1280))

                task_dim = int(getattr(configs, "linear_mix_task_dim", 128))
                hidden = int(getattr(configs, "linear_mix_hidden", 256))
                dropout = float(getattr(configs, "linear_mix_dropout", 0.0))
                use_ln = bool(getattr(configs, "linear_mix_use_layernorm", False))

                token_embed_table = self.esm2.embed_tokens.weight  # [V, H]

                raw_mask_token_ids = getattr(configs, "linear_mix_mask_token_ids", [0, 1, 2, 3, 29, 30, 31, 32])
                if raw_mask_token_ids is not None:
                    mask_token_ids = list(raw_mask_token_ids)
                    mask_token_ids = [int(x) for x in mask_token_ids]
                else:
                    mask_token_ids = None

                # 安全检查：ESM hidden size 要与 projector 期望一致
                if token_embed_table.size(1) != embed_size:
                    raise ValueError(
                        f"Embedding dim mismatch: ESM={token_embed_table.size(1)} vs configs.task_dimension={embed_size}"
                    )

                self.prompt_gen = MLPMixPrompt(
                    embed_weight=token_embed_table,
                    prompt_len=prompt_len,
                    num_tasks=num_tasks,
                    task_dim=task_dim,
                    hidden=hidden,
                    temperature=temperature,
                    mask_token_ids=mask_token_ids,
                    dropout=dropout,
                    use_layernorm=use_ln,
                )

            elif configs.task == 'random_from_embedding_table':
                self.prompt_layer_dict = nn.ParameterDict()
                num_tasks = configs.task_number
                prompt_len = configs.task_shape
                embed_size = configs.task_dimension
                token_embed_table = self.esm2.embed_tokens.weight

                for i in range(0, num_tasks):
                    self.prompt_layer_dict[str(i)] = nn.Parameter(
                        (from_sample_of_embeddings(token_embed_table)([prompt_len, embed_size], i)))

            elif configs.task == 'random':
                self.prompt_layer_dict = nn.ParameterDict({
                    f"{i}": nn.Parameter(torch.randn(configs.task_shape, configs.task_dimension))
                    for i in range(configs.encoder.prompt.num_tasks)
                })
                if configs.initialization_method == 'Uniform':
                    for key, param in self.prompt_layer_dict.items():
                        nn.init.uniform_(param, a=-0.5, b=0.5)
                elif configs.initialization_method == 'Gaussian':
                    # For the prompt layer dictionary, we initialize them using uniform distribution (same as in your previous code)
                    for key, param in self.prompt_layer_dict.items():
                        nn.init.normal_(param, mean=0.0, std=self.init_scale)

        self.configs = configs

        if self.configs.projector.projector_type == 'Co_attention_tranception':
            self.mlp = CoattentiontraceptionBlock(self.configs)

    def forward(self, x, task_ids,compute_saliency=False):
        features = self.esm2(x['input_ids'].to(device), repr_layers=[self.esm2.num_layers], task_ids=task_ids,
                             configs=self.configs)['representations'][self.esm2.num_layers]
        if self.configs.task == 'linear_mix':
            features_task, weights = self.prompt_gen(task_ids)


        if self.configs.projector.projector_type == 'Tranception':
            outputs = self.tranception_block(hidden_states=features, encoder_hidden_states=features_task)
        elif self.configs.projector.projector_type == 'Sequential_Bidirectional_Tranception':
            outputs = self.tranception_block(protein_states=features, task_states=features_task)
        else:
            B, L, D = features_task.shape


            outputs = self.mlp(features,features_task)

        return outputs


def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for para_name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            print(para_name)
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(model, logging):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)
    logging.info(
        f"trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_models_secondary_structure_ptm(configs):
    """
    Prepare the encoder model.

    Args:
        configs: A python box object containing the configuration options.
        logging: The logging object.

    Returns:
        The encoder model.
    """
    # Prepare the encoder.
    encoder = EncoderSSPTM( configs=configs)
    return encoder
