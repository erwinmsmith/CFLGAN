import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from pyspark.sql import SparkSession

# ===== Initialize SparkSession =====
spark = SparkSession.builder \
    .appName("PySpark-OpenFedLLM Integration") \
    .getOrCreate()

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

for round in tqdm(range(fed_args.num_rounds)):
    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")

    # ===== Train local models in parallel using PySpark =====
    def train_client(client_id):
        if client_id not in clients_this_round:
            return (-1, None)  # -1 表示未训练，None 表示无更新

        set_peft_model_state_dict(model, global_dict)  # 同步全局模型到本地模型
        sub_dataset = get_dataset_this_round(local_datasets[client_id], round, fed_args, script_args)
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)
        training_args = get_training_args(script_args, new_lr)

        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client_id],
            global_auxiliary=global_auxiliary,
        )

        results = trainer.train()
        local_dict = copy.deepcopy(get_peft_model_state_dict(model))  # 获取本地模型状态
        return (results.training_loss, local_dict)

    # 使用 PySpark 的 parallelize 和 map 来并行训练客户端模型
    client_ids = list(range(fed_args.num_clients))
    client_rdd = spark.sparkContext.parallelize(client_ids)
    client_results = client_rdd.map(train_client).collect()

    # 更新训练损失和本地模型状态
    for client_id, (loss, local_dict) in enumerate(client_results):
        if loss != -1:
            training_loss[client_id].append(loss)
            local_dict_list[client_id] = local_dict

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    set_peft_model_state_dict(model, global_dict)   # 更新全局模型

    # ===== Save the model =====
    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

# ===== Save the final model =====
trainer.save_model(os.path.join(script_args.output_dir, "final_model"))
np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

# ===== Stop SparkSession =====
spark.stop()