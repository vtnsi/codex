import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import random
import numpy as np
import glob
import math

def balanced_train(df:pd.DataFrame, interaction_indices:dict, extract_combination=None, output_dir=''):
    df_selected = None
    df_selected_filename = None
    
    for combination in interaction_indices:
        print(combination, type(combination))
        combination_dict = interaction_indices[combination]
        interaction_least_freq = min(combination_dict, key=lambda x: len(set(combination_dict[x])))
        interaction_max_freq = max(combination_dict, key=lambda x: len(set(combination_dict[x])))

        i = len(combination_dict)
        n = len(combination_dict[interaction_least_freq])
        ni = n*i
        
        sample_others = n
        sample_interaction = n
        if sample_others < 1:
            sample_others = 1

        #OTHER
        df_other = pd.DataFrame()
        for interaction_other in combination_dict:
            interaction_other_ids = combination_dict[interaction_other]
            
            count_other = len(interaction_other_ids)
            sample_others_max_possible = min([sample_others, count_other])

            other_ids_sampled = random.sample(interaction_other_ids, sample_others_max_possible)
            print("{}: {} sampled among interaction {}. Difference of {} from target.".format(combination, sample_others_max_possible, interaction_other, sample_others_max_possible-sample_others))
            
            df_other = pd.concat((df_other, df.loc[df.index.isin(other_ids_sampled)]), axis=0)

        
        df_skewed = pd.concat([df_other],axis=0)
        df_skewed.sample(len(df_skewed))
        filename = 'training_skew_<{} {}>-{}.csv'.format(combination, 'balanced', ni)
        df_skewed.to_csv(os.path.join(output_dir, filename))
        print('Total num in balanced DF:', df_skewed.shape)
        print()

        if extract_combination is not None and combination == extract_combination:
            df_selected = df_skewed
            df_selected_filename = filename
            print("Chosen: {}".format(combination))

    print("Created {} training sets for each combination (t=2).".format(len(interaction_indices)))
    return df_selected, df_selected_filename

def output_json_readable(json_obj:dict, print_json=False, write_json=False, file_path='', sort=False):
    '''
        Formats JSON object to human-readable format with print/save options.
    '''
    json_str = json.dumps(json_obj, sort_keys=sort, indent=4,separators=(',', ': '))
    
    if print_json:
        print(json_str)
        
    if write_json:
        if file_path == '':
            file_path = 'output_0{}.json'.format(len(glob.glob('output_0*.json')))
        with open(file_path, 'w') as f:
            f.write(json_str)
    
    return json_obj

def get_combinations(selected_features, universe):
    for feature_i in selected_features:
        

    return

def interaction_indices_t2(df:pd.DataFrame, selected_features=[], universe=None):
    try:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    except:
        print("Already dropped Unnamed")

    if len(selected_features) == 0:
        selected_features = df.columns.tolist()

    interaction_lengths = []
    indices_per_interaction = {}
    seen_combos = set()

    for feature_i in selected_features:
        values = sorted(df[feature_i].unique())
        
        for feature_j in selected_features:
            if feature_i == feature_j:
                continue
            
            if (feature_j, feature_i) not in seen_combos:
                seen_combos.add((feature_i, feature_j))
            else:
                continue
                
            values_j = sorted(df[feature_j].unique())

            combination = '({}*{})'.format(feature_i, feature_j)#str((feature_i, feature_j))
            indices_per_interaction[combination] = {}

            for value_i in values:
                for value_j in values_j:
                    interaction = '({}*{})'.format(value_i, value_j)

                    df_feat_val_i = df[df[feature_i] == value_i]
                    skew_df = df_feat_val_i[df_feat_val_i[feature_j] == value_j]
                    skew_indices = skew_df.index.tolist()
                    indices_per_interaction[combination][interaction] = skew_indices
                    interaction_lengths.append(len(skew_indices))
    
    return indices_per_interaction, list(indices_per_interaction.keys())

def skew_dataset_relative(df:pd.DataFrame, interaction_indices:dict,
                          extract_combination='', skew_level=None, output_dir=''):
    
    df_selected=None; df_selected_filename=None; combo_id_selected=None; interaction_id_selected=None
    ni = None


    combination_dict = interaction_indices[extract_combination]
    interaction_least_freq = min(combination_dict, key=lambda x: len(set(combination_dict[x])))
    interaction_max_freq = max(combination_dict, key=lambda x: len(set(combination_dict[x])))
    least_freq_interaction_ids = combination_dict[interaction_least_freq]
    most_freq_interaction_ids = combination_dict[interaction_max_freq]
    
    i = len(combination_dict)
    n = len(least_freq_interaction_ids)
    m = len(most_freq_interaction_ids)

    ni = int(n*i)

    x_hi = 0.9
    x_lo = 1/i
    x_mod = np.mean([x_hi, x_lo])
    if skew_level == 'high':
        x = x_hi
    elif skew_level == 'moderate':
        x = x_mod
    elif skew_level == 'low':
        x = x_lo
    elif skew_level == 'baseline':
        x = 1.0

    xni = int(x*ni)

    print("TRAINING DF SIZE DESIRED FOR {}: {}".format(extract_combination, ni))

    if skew_level == 'baseline':
        df_selected = df.sample(int(ni))
        df_selected_filename = 'training_skew_<{} {}>-{}_random.csv'.format(extract_combination, interaction_max_freq, 'baseline')
        combo_id_selected = f'{extract_combination}-{skew_level}'
        interaction_id_selected = interaction_max_freq

        return df_selected, df_selected_filename, combo_id_selected, interaction_id_selected

    print("Sampling interaction of interest. Desiring xni = {} of interaction {} from actual={}".format(xni, interaction_max_freq, m))
    sample_interaction_max = int(min(np.floor([xni, m])))
    print("Choosing {} samples.".format(sample_interaction_max))
    df_interaction = df.loc[df.index.isin(most_freq_interaction_ids)].sample(sample_interaction_max)
    print("{}: Sampled {} among MOST FREQ interaction, {} samples. Difference of {} from original target.".format(extract_combination, len(df_interaction), interaction_max_freq, sample_interaction_max-xni))
    
    print("Sampling other interactions in combination with remainder {} among {} other interactions.".format(ni-sample_interaction_max, i-1))
    sample_others = int(np.ceil((ni - sample_interaction_max)/(i-1)))
    if sample_others < 1:
        sample_others = 1
    
    #OTHER
    df_other = pd.DataFrame()
    for interaction_other in combination_dict:
        if interaction_max_freq == interaction_other:
            continue

        interaction_other_ids = combination_dict[interaction_other]
        m_other = len(interaction_other_ids)
        sample_others_max = int(min([sample_others, m_other]))

        other_ids_sampled = random.sample(interaction_other_ids, sample_others_max)
        df_other = pd.concat((df_other, df.loc[df.index.isin(other_ids_sampled)]), axis=0)
        print("{}: Sampled {} among interaction {}. Difference of {} from original target.".format(extract_combination, len(other_ids_sampled), interaction_other, sample_others_max-((sample_interaction_max-xni)/i-1)))

    filename = 'training_skew_<{} {}>-{}.csv'.format(extract_combination, interaction_max_freq, skew_level)
    df_skewed = pd.concat([df_interaction, df_other],axis=0)
    df_skewed = df_skewed.sample(len(df_skewed))
    df_skewed.to_csv(os.path.join(output_dir, filename))
    df_selected = df_skewed
    df_selected_filename = filename
    combo_id_selected = f'{extract_combination}-{skew_level}'
    interaction_id_selected = interaction_max_freq

    print('Total num in skewed DF towards {} {}:'.format(extract_combination, interaction_max_freq), df_skewed.shape)
    print()
        
    print("CHOOSING: {}. {} samples".format(extract_combination, len(df_selected)))
    print()

    print(skew_level, df_selected.head())
    print("Created {} training sets for each combination (t=2).".format(len(interaction_indices)))

    if m < xni:
        print('!!!')
    print("???")
    print(extract_combination)
    print(skew_level)
    print("xni, m", xni, m)
    print("min ", sample_interaction_max)
    print("PROP: ", sample_interaction_max/len(df_selected))
            
    return df_selected, df_selected_filename, combo_id_selected, interaction_id_selected

def main():

    return

if __name__ == '__main__':
    main()

