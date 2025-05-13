import random

def split_dataset(fed_args, script_args, dataset):
    dataset = dataset.shuffle(seed=script_args.seed)  # Shuffle the dataset
    local_datasets = []

    if fed_args.split_strategy == "iid":
        # 均匀划分数据集（iid）
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    elif fed_args.split_strategy == "non-iid":
        # 按标签分组（non-iid）
        # 假设数据集中有一个 'label' 字段
        label_groups = defaultdict(list)
        for idx, item in enumerate(dataset):
            label = item['label']
            label_groups[label].append(idx)
        
        # 将标签组分配给客户端
        num_clients = fed_args.num_clients
        label_group_list = list(label_groups.values())
        num_label_groups = len(label_group_list)
        
        for client_id in range(num_clients):
            client_data_indices = []
            # 将每个标签组分配给一个客户端
            for i in range(client_id, num_label_groups, num_clients):
                client_data_indices.extend(label_group_list[i])
            
            # 创建客户端数据集
            client_data = dataset.select(client_data_indices)
            local_datasets.append(client_data)
    
    return local_datasets

def get_dataset_this_round(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round