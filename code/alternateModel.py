# %%
# Importing libraries
import time
import argparse
import pickle
import os
import gc
import re

import pandas as pd
import numpy as np
import lightgbm
# %%
# Defining datafolder and importing dataframe
train_datafile = '../Dataset/VU_DM_data/fixed_training_set_VU_DM.csv'
df_train = pd.read_csv(train_datafile)
test_datafile = '../Dataset/VU_DM_data/fixed_test_set_VU_DM.csv'
df_test = pd.read_csv(test_datafile)

df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_test = df_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# %%
# Defining Functions
def normalize_features(input_df, group_key, target_column, take_log10=False):

    # for numerical stability
    epsilon = 1e-4
    if take_log10:
        input_df[target_column] = np.log10(input_df[target_column] + epsilon)
    methods = ["mean", "std"]

    df = input_df.groupby(group_key).agg({target_column: methods})

    df.columns = df.columns.droplevel()
    col = {}
    for method in methods:
        col[method] = target_column + "_" + method

    df.rename(columns=col, inplace=True)
    df_merge = input_df.merge(df.reset_index(), on=group_key)
    df_merge[target_column + "_norm_by_" + group_key] = (
        df_merge[target_column] - df_merge[target_column + "_mean"]
    ) / df_merge[target_column + "_std"]
    df_merge = df_merge.drop(labels=[col["mean"], col["std"]], axis=1)

    gc.collect()
    return df_merge
# Add new columns to the dataframe
def aggregated_features_single_column(
    in_data,
    key_for_grouped_by="prop_id",
    target_column="price_usd",
    agg_methods=["mean", "median", "min", "max"],
    transform_methods={"mean": ["substract"]},
):
    df = in_data.groupby(key_for_grouped_by).agg({target_column: agg_methods})

    if isinstance(key_for_grouped_by, list):
        str_key_for_grouped_by = "|".join(key_for_grouped_by)
    else:
        str_key_for_grouped_by = key_for_grouped_by

    df.columns = df.columns.droplevel()
    col = {}
    for method in agg_methods:
        col[method] = (
            method.upper() + "(" + str_key_for_grouped_by + ", " + target_column + ")"
        )

    df.rename(columns=col, inplace=True)

    in_data = in_data.merge(df.reset_index(), on=key_for_grouped_by)
    for method_name in transform_methods:
        for applying_function in transform_methods[method_name]:
            function_data = in_data[
                method_name.upper()
                + "("
                + str_key_for_grouped_by
                + ", "
                + target_column
                + ")"
            ]
            column_data = in_data[target_column]
            if applying_function == "substract":
                result = column_data - function_data
            elif applying_function == "divide":
                result = column_data / function_data
            else:
                continue
            in_data[
                applying_function.upper()
                + "("
                + target_column
                + ", "
                + method_name.upper()
                + ")"
            ] = result
    gc.collect()

    return in_data

def preprocess_training_data(orig_data, kind="train", use_ndcg_choices=False):

    print("Preprocessing training data....")
    gc.collect()
    data_for_training = orig_data

    target_column = "target"

    if kind == "train":
        conditions = [
            data_for_training["rank"] == 1,
            data_for_training["rank"] == 2,
        ]
        choices = [1, 2]
        data_for_training[target_column] = np.select(conditions, choices, default=0)

    threshold = 0.9

    # do not normalize 2 times with take_log10
    data_for_training = normalize_features(
        data_for_training,
        group_key="srch_id",
        target_column="price_usd",
        take_log10=True,
    )

    data_for_training = aggregated_features_single_column(
        data_for_training, "prop_id", "price_usd", ["mean"]
    )
    data_for_training = aggregated_features_single_column(
        data_for_training, "prop_id", "total_comp", ["mean"]
    )
    data_for_training = aggregated_features_single_column(
        data_for_training, "prop_id", "total_comp_rate", ["mean"]
    )
    data_for_training = aggregated_features_single_column(
        data_for_training, "prop_id", "rel_comp_inv", ["mean"]
    )
    data_for_training = aggregated_features_single_column(
        data_for_training, "prop_id", "avg_comp_rate_diff", ["mean"]
    )

    data_for_training = normalize_features(
        data_for_training, group_key="prop_id", target_column="price_usd"
    )
    data_for_training = normalize_features(
        data_for_training, group_key="prop_id", target_column="total_comp"
    )
    data_for_training = normalize_features(
        data_for_training, group_key="prop_id", target_column="total_comp_rate"
    )
    data_for_training = normalize_features(
        data_for_training, group_key="prop_id", target_column="rel_comp_inv"
    )
    data_for_training = normalize_features(
        data_for_training, group_key="prop_id", target_column="avg_comp_rate_diff"
    )
    data_for_training = normalize_features(
        data_for_training, group_key="srch_id", target_column="prop_starrating"
    )

    data_for_training = aggregated_features_single_column(
        data_for_training,
        key_for_grouped_by="srch_id",
        target_column="prop_starrating",
        agg_methods=["mean"],
        transform_methods={"mean": ["substract"]},
    )
    # data_for_training = aggregated_features_single_column(
    #     data_for_training,
    #     key_for_grouped_by="srch_id",
    #     target_column="prop_location_score2",
    #     agg_methods=["mean"],
    #     transform_methods={"mean": ["substract"]},
    # )
    data_for_training = aggregated_features_single_column(
        data_for_training,
        key_for_grouped_by="srch_id",
        target_column="prop_location_score1",
        agg_methods=["mean"],
        transform_methods={"mean": ["substract"]},
    )
    data_for_training = aggregated_features_single_column(
        data_for_training,
        key_for_grouped_by="srch_destination_id",
        target_column="price_usd",
        agg_methods=["mean"],
        transform_methods={"mean": ["substract"]},
    )
    data_for_training = aggregated_features_single_column(
        data_for_training,
        key_for_grouped_by="srch_id",
        target_column="prop_review_score",
        agg_methods=["mean"],
        transform_methods={"mean": ["substract"]},
    )
    data_for_training = aggregated_features_single_column(
        data_for_training,
        key_for_grouped_by="srch_id",
        target_column="promotion_flag",
        agg_methods=["mean"],
        transform_methods={"mean": ["substract"]},
    )

    # NOTE: has to be done after aggregated_features_single_column
    data_for_training = data_for_training.sort_values("srch_id")

    data_for_training = normalize_features(
        data_for_training, group_key="srch_id", target_column="prop_starrating"
    )
    data_for_training = normalize_features(
        data_for_training, group_key="srch_id", target_column="prop_location_score2"
    )
    data_for_training = normalize_features(
        data_for_training, group_key="srch_id", target_column="prop_location_score1"
    )
    data_for_training = normalize_features(
        data_for_training, group_key="srch_id", target_column="prop_review_score"
    )

    gc.collect()
    if kind == "train":
        y = data_for_training[target_column].values
    else:
        y = None

    training_set_only_metrics = ["rank", "gross_bookings_usd"]
    columns_to_remove = [
        "target",
        target_column,
    ] + training_set_only_metrics
    columns_to_remove = [
        c for c in columns_to_remove if c in data_for_training.columns.values
    ]
    data_for_training = data_for_training.drop(labels=columns_to_remove, axis=1)
    return data_for_training, y

def input_estimated_position(training_data, srch_id_dest_id_dict):
    training_data = training_data.merge(
        srch_id_dest_id_dict, how="left", on=["srch_destination_id", "prop_id"]
    )
    print(training_data.head())
    return training_data

def remove_columns(x1, ignore_column=["srch_id", "prop_id", "position", "random_bool"]):
    ignore_column = [c for c in ignore_column if c in x1.columns.values]
    # print('Dropping columns: {}'.format(ignore_column))
    # ignore_column_numbers = [x1.columns.get_loc(x) for x in ignore_column]
    x1 = x1.drop(labels=ignore_column, axis=1)
    # print('Columns after dropping: {}'.format(x1.columns.values))
    return x1

def split_train_data(data_for_training, y, val_start=0, val_end=0):

    x1 = pd.concat([data_for_training[0:val_start], data_for_training[val_end:]])
    y1 = np.concatenate((y[0:val_start], y[val_end:]), axis=0)
    x2 = data_for_training[val_start:val_end]
    y2 = y[val_start:val_end]

    srch_id_dest_id_dict = x1.loc[x1["random_bool"] == 0]

    # estimated position calculation
    srch_id_dest_id_dict = x1.loc[x1["random_bool"] == 0]
    srch_id_dest_id_dict = x1.groupby(["srch_destination_id", "prop_id"]).agg(
        {"position": "mean"}
    )
    srch_id_dest_id_dict = srch_id_dest_id_dict.rename(
        index=str, columns={"position": "estimated_position"}
    ).reset_index()
    srch_id_dest_id_dict["srch_destination_id"] = (
        srch_id_dest_id_dict["srch_destination_id"].astype(str).astype(int)
    )
    srch_id_dest_id_dict["prop_id"] = (
        srch_id_dest_id_dict["prop_id"].astype(str).astype(int)
    )
    srch_id_dest_id_dict["estimated_position"] = (
        1 / srch_id_dest_id_dict["estimated_position"]
    )
    x1 = input_estimated_position(x1, srch_id_dest_id_dict)
    x2 = input_estimated_position(x2, srch_id_dest_id_dict)

    groups = x1["srch_id"].value_counts(sort=False).sort_index()
    eval_groups = x2["srch_id"].value_counts(sort=False).sort_index()
    len(eval_groups), len(x2), len(x1), len(groups)

    x1 = remove_columns(x1)
    x2 = remove_columns(x2)
    return (x1, x2, y1, y2, groups, eval_groups, srch_id_dest_id_dict)

def get_categorical_column(x1):
    categorical_features = [
        "weekday",
        "prop_country_id",
        "site_id",
        "visitor_location_country_id",
    ]
    categorical_features = [c for c in categorical_features if c in x1.columns.values]
    categorical_features_numbers = [x1.columns.get_loc(x) for x in categorical_features]
    return categorical_features_numbers

def train_model(
    x1, x2, y1, y2, groups, eval_groups, lr, method, output_dir='./', name_of_model=None
):
    if not name_of_model:
        name_of_model = str(int(time.time()))

    categorical_features_numbers = get_categorical_column(x1)
    clf = lightgbm.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=5000,
        learning_rate=lr,
        max_position=5,
        label_gain=[0, 1, 2],
        random_state=69,
        seed=69,
        boosting=method,
    )
    gc.collect()

    print("Training on train set with columns: {}".format(x1.columns.values))
    clf.fit(
        x1,
        y1,
        eval_set=[(x1, y1), (x2, y2)],
        eval_group=[groups, eval_groups],
        group=groups,
        eval_at=5,
        verbose=50,
        early_stopping_rounds=500,
        categorical_feature=categorical_features_numbers,
    )
    gc.collect()
    pickle.dump(clf, open(os.path.join(output_dir, "model.dat"), "wb"))
    return clf

def predict(name_of_model, test_data, srch_id_dest_id_dict, output_dir='./'):

    gc.collect()

    model = pickle.load(open(os.path.join(output_dir, "model.dat"), "rb"))

    test_data = test_data.copy()
    test_data = input_estimated_position(test_data, srch_id_dest_id_dict)

    test_data_srch_id_prop_id = test_data[["srch_id", "prop_id"]]

    test_data = remove_columns(test_data)

    categorical_features_numbers = get_categorical_column(test_data)

    print("Predicting on train set with columns: {}".format(test_data.columns.values))
    kwargs = {}
    kwargs = {"categorical_feature": categorical_features_numbers}

    predictions = model.predict(test_data, **kwargs)
    test_data_srch_id_prop_id["prediction"] = predictions
    del test_data
    gc.collect()

    test_data_srch_id_prop_id = test_data_srch_id_prop_id.sort_values(
        ["srch_id", "prediction"], ascending=False
    )
    print("Saving predictions into submission.csv")
    test_data_srch_id_prop_id[["srch_id", "prop_id"]].to_csv(
        os.path.join(output_dir, "new_submission5_5_500.csv"), index=False
    )

# %%
# Starting Model Training
name_of_model = str(int(time.time()))

training_data = df_train
training_data, y = preprocess_training_data(training_data)

method = "goss"
validation_num = 150000
lr = 0.12

for i in range(0, 1):
    val_start = i * validation_num
    val_end = (i + 1) * validation_num
    x1, x2, y1, y2, groups, eval_groups, srch_id_dest_id_dict = split_train_data(
    training_data, y, val_start, val_end
    )
    model = train_model(
    x1, x2, y1, y2, groups, eval_groups, lr, method, './submissions/', name_of_model
    )
    test_data = df_test
    test_data, _ = preprocess_training_data(test_data, kind="test")
    predict(name_of_model, test_data, srch_id_dest_id_dict, './')
    print("Submit the predictions file submission.csv to kaggle")
# %%
