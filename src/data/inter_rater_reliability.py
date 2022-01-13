import pandas as pd
import numpy as np
import krippendorff
import pdb

annotations_dir = "./subset_annotation"
generation_dir = "./generation_annotation"

if __name__ == "__main__":
    mode = "generation"

    if mode == "annotation":
        df_1 = pd.read_csv(f"{annotations_dir}/gpt3_annot_1.csv")[["startphrase", "ending1", "ending2", "difficulty", "type"]]
        df_1["difficulty"] = df_1["difficulty"].fillna("easy")
        df_2 = pd.read_csv(f"{annotations_dir}/gpt3_annot_2.csv")[["startphrase", "ending1", "ending2", "difficulty", "type"]]
        df_2["difficulty"] = df_2["difficulty"].fillna("easy")
        df_3 = pd.read_csv(f"{annotations_dir}/gpt3_annot_3.csv")[["startphrase", "ending1", "ending2", "difficulty", "type"]]
        df_3["difficulty"] = df_3["difficulty"].fillna("easy")
        

        df_combined_difficulty = pd.DataFrame({"annot1_diff": df_1["difficulty"], "annot2_diff": df_2["difficulty"], "annot3_diff": df_3["difficulty"]})
        df_combined_difficulty['majority'] = df_combined_difficulty.mode(axis=1)[0]
        print("difficulty votes:", df_combined_difficulty["majority"].value_counts())

        df_combined_type = pd.DataFrame({"annot1": df_1["type"], "annot2": df_2["type"], "annot3": df_3["type"]})
        df_combined_type['majority'] = df_combined_type.mode(axis=1)[0]
        print("type votes:", df_combined_type["majority"].value_counts())

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

        df_two_agree_difficulty.to_csv("two_disagree_difficulty_gpt3.csv", index=False)
        df_all_disagree_type.to_csv("all_disagree_type_gpt3.csv", index=False)
        df_two_disagree_type.to_csv("two_disagree_type_gpt3.csv", index=False)
    elif mode == "generation":
        df_1 = pd.read_csv(f"{generation_dir}/annot_1.csv")
        df_2 = pd.read_csv(f"{generation_dir}/annot_2.csv")
        df_3 = pd.read_csv(f"{generation_dir}/annot_3.csv")
        df_combined_score = pd.DataFrame({"annot1_diff": df_1["valid"], "annot2_diff": df_2["valid"], "annot3_diff": df_3["valid"]})
        df_combined_score['majority'] = df_combined_score.mode(axis=1)[0]
        print(df_combined_score["majority"].value_counts())
        alpha = krippendorff.alpha([df_1["valid"], df_2["valid"], df_3["valid"]])
        print("alpha: ", alpha)

        contradict = 0
        one_correct = 0
        one_wrong = 0
        all_correct = 0
        valid_lst = list(df_combined_score["majority"])
        for i in range(0, len(df_combined_score), 4):
            annot = []
            for j in range(0, 4):
                annot.append(valid_lst[i + j])
            
            if 1.0 in annot:
                one_correct += 1
            if 0.0 in annot:
                one_wrong += 0
            if 1.0 in annot and 0.0 in annot:
                contradict += 1
            if all([1 if x == 1.0 else 0 for x in annot]):
                all_correct += 1
        
        print("contradictions: ", contradict/(len(valid_lst)/4))
        print("one correct: ", one_correct/(len(valid_lst)/4))
        print("all correct: ", all_correct/(len(valid_lst)/4))
