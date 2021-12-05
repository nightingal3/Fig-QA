import pandas as pd
import krippendorff
import pdb

annotations_dir = "./subset_annotation"

if __name__ == "__main__":
    df_1 = pd.read_csv(f"{annotations_dir}/annot_1.csv")[["startphrase", "ending1", "ending2", "difficulty", "type"]]
    df_1["difficulty"] = df_1["difficulty"].fillna("easy")
    df_2 = pd.read_csv(f"{annotations_dir}/annot_2.csv")[["startphrase", "ending1", "ending2", "difficulty", "type"]]
    df_2["difficulty"] = df_2["difficulty"].fillna("easy")
    df_3 = pd.read_csv(f"{annotations_dir}/annot_3.csv")[["startphrase", "ending1", "ending2", "difficulty", "type"]]
    df_3["difficulty"] = df_3["difficulty"].fillna("easy")

    combined_arr = pd.concat([df_1, df_2, df_3], axis=1)

    difficulty_arr = pd.concat([df_1["difficulty"], df_2["difficulty"], df_3["difficulty"]], axis=1).replace({"easy": 0, "hard": 1}).to_numpy()
    alpha_d = krippendorff.alpha(difficulty_arr)

    type_arr = pd.concat([df_1["type"], df_2["type"], df_3["type"]], axis=1).replace({"obj": 0, "soc": 1, "vis": 2, "cul": 3}).to_numpy()
    alpha_t = krippendorff.alpha(type_arr)
    
    all_disagree_difficulty = combined_arr["difficulty"].T.apply(set).map(len) == 3 # None
    two_disagree_difficulty = combined_arr["difficulty"].T.apply(set).map(len) == 2

    df_two_agree_difficulty = combined_arr[two_disagree_difficulty]

    all_disagree_type = combined_arr["type"].T.apply(set).map(len) == 3 # None
    two_disagree_type = combined_arr["type"].T.apply(set).map(len) == 2

    df_all_disagree_type = combined_arr[all_disagree_type]
    df_two_disagree_type = combined_arr[two_disagree_type]
    
    print("alpha score for difficulty", alpha_d)
    print("alpha score for type", alpha_t)

    df_two_agree_difficulty.to_csv("two_disagree_difficulty.csv", index=False)
    df_all_disagree_type.to_csv("all_disagree_type.csv", index=False)
    df_two_disagree_type.to_csv("two_disagree_type.csv", index=False)