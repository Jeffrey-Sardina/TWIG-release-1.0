import pandas as pd
from utils import gather_data
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader

'''
===============
Reproducibility
===============
'''
torch.manual_seed(42)

def get_adj_data(valid_triples_map):
    ents_to_triples = {} # entity to all relevent data
    for t_idx in valid_triples_map:
        s, p, o = valid_triples_map[t_idx]
        if not s in ents_to_triples:
            ents_to_triples[s] = set()
        if not o in ents_to_triples:
            ents_to_triples[o] = set()
        ents_to_triples[s].add(t_idx)
        ents_to_triples[o].add(t_idx)
    return ents_to_triples

def get_twm_data_augment(dataset_name,
    exp_dir,
    exp_id=None,
    incl_hps=True,
    incl_global_struct=False,
    incl_neighbour_structs=False,
    struct_source='train',
    incl_mrr=False,
    randomise=True):
    
    overall_results, \
        triples_results, \
        grid, \
        valid_triples_map, \
        graph_stats = gather_data(dataset_name, exp_dir)
    ents_to_triples = get_adj_data(valid_triples_map)

    all_data = []
    if exp_id is not None:
        iter_over = [str(exp_id)]
    else:
        iter_over = list(triples_results.keys())
        if randomise:
            random.shuffle(iter_over)
    global_struct = {}
    if incl_global_struct:
        max_rank = len(graph_stats['all']['degrees']) # = num nodes
        global_struct["max_rank"] = max_rank
        # percentiles_wanted = [0, 5, 10, 25, 50, 75, 90, 95, 100]
        # for p in percentiles_wanted:
        #     global_struct[f'node_deg_p_{p}'] = graph_stats[struct_source]['percentiles'][p]
        #     global_struct[f'rel_freq_p_{p}'] = graph_stats[struct_source]['total_rel_degree_percentiles'][p]

    for exp_id in iter_over:
        mrr = overall_results[exp_id]['mrr']
        hps = grid[exp_id]
        for triple_idx in valid_triples_map:
            s, p, o = valid_triples_map[triple_idx]

            s_deg = graph_stats[struct_source]['degrees'][s] \
                if s in graph_stats[struct_source]['degrees'] else 0
            o_deg = graph_stats[struct_source]['degrees'][o] \
                if o in graph_stats[struct_source]['degrees'] else 0
            p_freq = graph_stats[struct_source]['pred_freqs'][p] \
                if p in graph_stats[struct_source]['pred_freqs'] else 0

            s_p_cofreq = graph_stats[struct_source]['subj_relationship_degrees'][(s,p)] \
                if (s,p) in graph_stats[struct_source]['subj_relationship_degrees'] else 0
            o_p_cofreq = graph_stats[struct_source]['obj_relationship_degrees'][(o,p)] \
                if (o,p) in graph_stats[struct_source]['obj_relationship_degrees'] else 0
            s_o_cofreq = graph_stats[struct_source]['subj_obj_cofreqs'][(s,o)] \
                if (s,o) in graph_stats[struct_source]['subj_obj_cofreqs'] else 0

            head_rank = triples_results[exp_id][triple_idx]['head_rank']
            tail_rank = triples_results[exp_id][triple_idx]['tail_rank']
            
            data = {}
            if incl_global_struct:
                for key in global_struct:
                    data[key] = global_struct[key]

            data['s_deg'] = s_deg
            data['o_deg'] = o_deg
            data['p_freq'] = p_freq
            data['s_p_cofreq'] = s_p_cofreq
            data['o_p_cofreq'] = o_p_cofreq
            data['s_o_cofreq'] = s_o_cofreq
            data['head_rank'] = head_rank
            data['tail_rank'] = tail_rank

            if incl_neighbour_structs:
                for target in s, o: 
                    target_name = 's' if target == s else 'o'
                    neighbour_nodes = {}
                    neighbour_preds = {}
                    for t_idx in ents_to_triples[target]:
                        t_s, t_p, t_o = valid_triples_map[t_idx]
                        ent = t_s if target != t_s else t_o
                        if not t_p in neighbour_preds:
                            neighbour_preds[t_p] = graph_stats[struct_source]['pred_freqs'][t_p]
                        if not ent in neighbour_nodes:
                            neighbour_nodes[ent] = graph_stats[struct_source]['degrees'][ent]

                    data[f'{target_name} min deg neighbnour'] = np.min(list(neighbour_nodes.values()))
                    data[f'{target_name} max deg neighbnour'] = np.max(list(neighbour_nodes.values()))
                    data[f'{target_name} mean deg neighbnour'] = np.mean(list(neighbour_nodes.values()))
                    data[f'{target_name} num neighbnours'] = len(neighbour_nodes)

                    data[f'{target_name} min freq rel'] = np.min(list(neighbour_preds.values()))
                    data[f'{target_name} max freq rel'] = np.max(list(neighbour_preds.values()))
                    data[f'{target_name} mean freq rel'] = np.mean(list(neighbour_preds.values()))
                    data[f'{target_name} num rels'] = len(neighbour_preds)

            if incl_hps:
                for key in hps:
                    data[key] = hps[key]
            if incl_mrr:
                data['mrr'] = mrr
            all_data.append(data)

    '''
    We now want to make this to instead of head and tail rank independently,
    we just have one 'rank' column
    '''
    rank_data = []
    for data_dict in all_data:
        head_data = {key: data_dict[key] for key in data_dict}
        del head_data['tail_rank']
        rank = head_data['head_rank']
        del head_data['head_rank']
        head_data['rank'] = rank
        head_data['is_head'] = 1

        tail_data = {key: data_dict[key] for key in data_dict}
        del tail_data['head_rank']
        rank = tail_data['tail_rank']
        del tail_data['tail_rank']
        tail_data['rank'] = rank
        tail_data['is_head'] = 0

        rank_data.append(head_data)
        rank_data.append(tail_data)

    rank_data_df = pd.DataFrame(rank_data)
    
    # move rank data to front
    is_head_col = rank_data_df.pop('is_head')
    rank_data_df.insert(0, 'is_head', is_head_col)

    return rank_data_df

def load_and_prep_twm_data_single_exp(rank_data_df,
                                      categorical_cols=[],
                                      normalisation='none'):
    target = "rank"
    test_ratio = 0.2
    normalisation = normalisation
    X_train, X_test, y_train, y_test, X, y = prepare_data(
        rank_data_df,
        categorical_cols=categorical_cols,
        target=target,
        test_ratio=test_ratio,
        normalisation=normalisation
    )
    return X_train, X_test, y_train, y_test, X, y


def prepare_data(df,
                    target='mrr',
                    categorical_cols = set(),
                    test_ratio = 0.2,
                    normalisation="none"):

    # separate target and data
    y = df[target]
    del df[target]
    X = df

    # one-hot code categorical vars: https://www.statology.org/pandas-get-dummies/
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.replace({None : 0})
    X_cols = list(X)

    # Make a train-test split
    # X_cols = poly.get_feature_names_out(X_cols)
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_ratio,
                                                        shuffle=True)
    X_train = pd.DataFrame(X_train, columns=X_cols)
    X_test = pd.DataFrame(X_test, columns=X_cols)
    X = pd.DataFrame(X, columns=X_cols)

    y_train = pd.DataFrame(y_train.T)
    y_test = pd.DataFrame(y_test.T)
    y = pd.DataFrame(y.T)

    # Normalise data
    numeric_columns = set(X_cols) - set(categorical_cols)
    assert normalisation in ('minmax', 'zscore', 'none')
    if normalisation != 'none':
        for col in numeric_columns:
            to_norm = X_train[col].to_numpy()
            if normalisation == 'minmax':
                col_min = to_norm.min()
                col_max = to_norm.max()
                if col_min == col_max:
                    if col_min > 0:
                        print('min = max; setting col to 1')
                        to_norm = to_norm / to_norm
                    else:
                        pass
            elif normalisation == 'zscore':
                col_mean = to_norm.mean()
                col_std = to_norm.std()
                if col_std > 0:
                    to_norm = (to_norm - col_mean) / col_std
                else:
                    to_norm = to_norm - col_mean
            X_train[col] = to_norm

    return X_train, X_test, y_train, y_test, X, y

def load_and_process_data(dataset_name, train_ids, test_id, try_load=True, allow_load_err=True):
    test_save_prefix = f'data_save/{dataset_name}_{test_id}'
    train_save_prefix = f'data_save/{dataset_name}_{"-".join(t for t in train_ids)}'
    loaded_data = False
    if try_load:
        try:
            print('Seeing if there are files to load...')
            training_x = torch.load(f'{train_save_prefix}--training_x')
            training_y = torch.load(f'{train_save_prefix}--training_y')
            testing_x = torch.load(f'{test_save_prefix}--testing_x')
            testing_y = torch.load(f'{test_save_prefix}--testing_y')
            print('Loaded fils!')
            loaded_data = True
        except:
            if not allow_load_err: raise

    if not loaded_data:
        print('Manually re-creating dataset...')
        training_x, training_y, testing_x, testing_y = load_and_process_tensors(
            dataset_name,
            train_ids,
            test_id,
            train_save_prefix=train_save_prefix,
            test_save_prefix=test_save_prefix
        )

    print('Configuring datasets on Torch')
    training_dataloader, testing_dataloader = create_torch_data_objects(
        training_x,
        training_y,
        testing_x,
        testing_y,
        train_ids
    )

    return training_dataloader, testing_dataloader

def load_and_process_tensors(dataset_name, train_ids, test_id, train_save_prefix=None, test_save_prefix=None):
    twm_hp_data = {}
    normalisation = 'none'
    exp_id = None
    incl_mrr = False
    incl_global_struct = False
    incl_neighbour_structs = True
    incl_hps = exp_id is None
    categorical_cols = ['loss', 'neg_samp'] if incl_hps else []

    # get training data
    for run in train_ids:
        exp_dir = f'../output/{dataset_name}/{dataset_name}-TWM-run{run}/'
        print(f'loading data from {exp_dir}')
        rank_data_df = get_twm_data_augment(dataset_name,
            exp_dir,
            exp_id=exp_id, #if none, get all
            incl_mrr=incl_mrr,
            incl_global_struct=incl_global_struct,
            incl_hps=incl_hps,
            incl_neighbour_structs=incl_neighbour_structs
        )
        X_train, X_test, y_train, y_test, X, y = load_and_prep_twm_data_single_exp(
            rank_data_df,
            categorical_cols=categorical_cols,
            normalisation=normalisation
        )
        twm_hp_data[f'{dataset_name}-{run}'] = {
            'X_train': X_train.fillna(0),
            'X_test': X_test.fillna(0),
            'y_train': y_train.fillna(0),
            'y_test': y_test.fillna(0),
            'X': X.fillna(0),
            'y': y.fillna(0)
        }

    twm_hp_data['Train-all'] = {}
    for key in twm_hp_data:
        if key == 'Train-all':
            continue
        for dataset in twm_hp_data[key]:
            print(f'concatenating data from twm_hp_data["{key}"]["{dataset}"]')
            if not dataset in twm_hp_data['Train-all']:
                twm_hp_data['Train-all'][dataset] = twm_hp_data[key][dataset]
            else:
                twm_hp_data['Train-all'][dataset] = pd.concat(
                    [
                        twm_hp_data['Train-all'][dataset],
                        twm_hp_data[key][dataset]
                    ],
                    ignore_index=True
                )

    # get testing data
    exp_dir = f'../output/{dataset_name}/{dataset_name}-TWM-run{test_id}/'
    print(f'loading testing data from {exp_dir}')
    rank_data_df = get_twm_data_augment(dataset_name,
        exp_dir,
        exp_id=exp_id, #if none, get all
        incl_mrr=incl_mrr,
        incl_global_struct=incl_global_struct,
        incl_hps=incl_hps,
        incl_neighbour_structs=incl_neighbour_structs
    )
    X_train, X_test, y_train, y_test, X, y = load_and_prep_twm_data_single_exp(
        rank_data_df,
        categorical_cols=categorical_cols,
        normalisation=normalisation
    )
    twm_hp_data[f'{dataset_name}-{test_id}'] = {
        'X_train': X_train.fillna(0),
        'X_test': X_test.fillna(0),
        'y_train': y_train.fillna(0),
        'y_test': y_test.fillna(0),
        'X': X.fillna(0),
        'y': y.fillna(0)
    }

    training_x = torch.Tensor(twm_hp_data['Train-all']['X'].to_numpy(dtype=np.float32))
    training_y = torch.Tensor(twm_hp_data['Train-all']['y'].to_numpy(dtype=np.float32))
    testing_x = torch.Tensor(twm_hp_data[f'{dataset_name}-{test_id}']['X'].to_numpy(dtype=np.float32))
    testing_y = torch.Tensor(twm_hp_data[f'{dataset_name}-{test_id}']['y'].to_numpy(dtype=np.float32))

    if train_save_prefix is not None:
        torch.save(training_x, f'{train_save_prefix}--training_x')
        torch.save(training_y, f'{train_save_prefix}--training_y')
    if test_save_prefix is not None:
        torch.save(testing_x, f'{test_save_prefix}--testing_x')
        torch.save(testing_y, f'{test_save_prefix}--testing_y')

    return training_x, training_y, testing_x, testing_y

def create_torch_data_objects(training_x, training_y, testing_x, testing_y, train_ids):
    print('calculating batch / chunk details')
    num_hp_settings = 1215
    num_train_runs = len(train_ids)
    divisor = num_hp_settings * num_train_runs
    assert training_y.shape[0] % divisor == 0, "Wrong divisor 1"
    training_batch_size = training_y.shape[0] // divisor # this is the number of corruptions we have per hp set

    num_hp_settings = 1215
    num_test_runs = 1
    divisor = num_hp_settings * num_test_runs
    assert testing_y.shape[0] % divisor == 0, "Wrong divisor 2"
    testing_batch_size = testing_y.shape[0] // divisor 

    training = TensorDataset(training_x, training_y)
    testing = TensorDataset(testing_x, testing_y)

    print(f'configuring batches; using training batch size {training_batch_size}')
    training_dataloader = DataLoader(
        training,
        batch_size=training_batch_size
    )
    print(f'configuring batches; using testing batch size {testing_batch_size}')
    testing_dataloader = DataLoader(
        testing,
        batch_size=testing_batch_size
    )

    return training_dataloader, testing_dataloader

def do_load(dataset_name, load_ids):
    test_id = load_ids[0]
    train_ids = load_ids[1:]
    return load_and_process_data(dataset_name, train_ids, test_id)
