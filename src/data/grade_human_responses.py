import pandas as pd
import os
import pdb

response_dir = "./human_responses"
test_sheet = "./filtered/test.csv"

if __name__ == "__main__":
    acc_all = []
    acc_confident_all = []
    total_correct = 0
    total_qs = 0
    total_correct_confident = 0
    total_confident = 0
    total_pairs = 0
    correct_pairs = 0
    reference_ans = pd.read_csv(test_sheet)

    incorrect_ans = []
    incorrect_ending1 = []
    incorrect_ending2 = []
    incorrect_labels = []
    for filename in os.listdir(response_dir):
        if not filename.endswith(".csv"):
            continue
        print(filename)
        human_ans = pd.read_csv(os.path.join(response_dir, filename))
        merged = human_ans.merge(reference_ans, how="left", on="startphrase")

        # some used different formatting
        if merged["answer"].max() == 2:
            merged["answer"] = merged["answer"].replace({1: 0, 2: 1})
        if "guess" in merged.columns:
            merged["notes"] = merged["guess"]
        if "notes" not in merged:
            confident = merged
        else:
            confident = merged.query("notes != notes") # people were instructed to write a note if unsure
        correct = merged.query('answer == labels')
        incorrect = merged.query('answer != labels')
        incorrect_phrase = list(incorrect["startphrase"])
        incorrect_option1 = list(incorrect["ending1_x"])
        incorrect_option2 = list(incorrect["ending2_x"])
        incorrect_label = list(incorrect["labels"])
    
        incorrect_ans.extend(incorrect_phrase)
        incorrect_ending1.extend(incorrect_option1)
        incorrect_ending2.extend(incorrect_option2)
        incorrect_labels.extend(incorrect_label)
        correct_confident = confident.query("answer == labels")
        correct_pairs += sum(correct["qid"].value_counts() == 2)
        total_pairs += sum(merged["qid"].value_counts() == 2)

        acc = len(correct)/len(merged)
        confident_acc = len(correct_confident)/len(confident)
        acc_all.append(acc)
        acc_confident_all.append(confident_acc)

        total_correct += len(correct)
        total_qs += len(merged)
        total_correct_confident += len(correct_confident)
        total_confident += len(confident)
        

    print("accuracies ", acc_all)
    print("accuracies (confident only)", acc_confident_all)
    print("overall", total_correct/total_qs)
    print("overall (confident)", total_correct_confident/total_confident)
    print("pair accuracies", correct_pairs/total_pairs)
    
    incorrect_cols = {"startphrase": incorrect_ans, "ending1": incorrect_ending1, "ending2": incorrect_ending2, "correct_label": incorrect_labels}
    df_incorrect = pd.DataFrame(incorrect_cols)
    df_incorrect.to_csv("incorrect_human.csv", index=False)