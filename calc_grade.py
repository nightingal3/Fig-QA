import csv
import re
import sys
from collections import defaultdict

ex_re = re.compile(r'(excuse|exemption|EX)')
quiz_re = re.compile(r'([0-9]) */ *([0-9])')
quiz_re2 = re.compile(r'([0-9\.]+)')
qid_re = re.compile(r' *\([0-9]+\)')
miss_re = re.compile(r'Miss')
quiz_columns = [6,7,8,10,11,12,14,15,17,18,19,20,21,22,23,24,25,27,28,29,30,31]
denoms =       [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
assign_columns = [9,13,26,32]
grade_map = {"A+": 1.0, "A": 0.96, "A-": 0.92, "B+": 0.88, "B": 0.85, "B-": 0.82, "C+": 0.78, "C": 0.75, "": 0.0}

def mod_quiz(qid, qstr):
  m = miss_re.match(qstr)
  if qstr == '' or m:
    return (0.0, '0 / {} (Missed)'.format(denoms[qid]))
  m = ex_re.search(qstr)
  if m:
    return (1000.0, qstr)
  m = quiz_re.match(qstr)
  if m:
    return (float(m.group(1))/denoms[qid], f'{m.group(1)} / {denoms[qid]}')
  m = quiz_re2.match(qstr)
  if m:
    return (float(m.group(1))/denoms[qid], f'{m.group(1)} / {denoms[qid]}')
  raise RuntimeError("bad string {}".format(qstr))
    
def drop_quiz(vals):
  qval, qstr = vals
  return 1000.0, "{} (Dropped)".format(qstr)

def threshold_grade(score):
  letter_grade = None
  if score > 96.95:
    letter_grade = "A+"
  elif score > 92.95:
    letter_grade = "A"
  elif score > 89.95:
    letter_grade = "A-"
  elif score > 86.95:
    letter_grade = "B+"
  elif score > 83.95:
    letter_grade = "B"
  elif score > 79.95:
    letter_grade = "B-"
  elif score > 76.95:
    letter_grade = "C+"
  elif score > 73.95:
    letter_grade = "C"
  elif score > 69.95:
    letter_grade = "C-"
  elif score > 66.95:
    letter_grade = "D+"
  elif score > 63.95:
    letter_grade = "D"
  elif score > 59.95:
    letter_grade = "D-"
  else:
    letter_grade = "F"
    # raise RuntimeError("score too low {}".format(score))
  return letter_grade

letters = defaultdict(lambda: 0)
andrews = {}

def calc_and_print_grades(name, andrew_id, quizzes, ass1_grade, ass2_grade, ass3_grade, project_grade):
  ids = list(range(len(quizzes)))
  counted_quizzes = [x for x in quizzes if x != 1000.0]
  ids.sort(key=lambda x: quizzes[x][0])
  quizzes[ids[0]] = drop_quiz(quizzes[ids[0]])
  quizzes[ids[1]] = drop_quiz(quizzes[ids[1]])
  quizzes[ids[2]] = drop_quiz(quizzes[ids[2]])
  skipped = [ids[0], ids[1], ids[2]]
  counted_quizzes = [x for x, y in quizzes if x != 1000.0]
  quiz_grade = sum(counted_quizzes)/len(counted_quizzes)

  print("\n***** Grade Report for {} *****\n".format(name))
  print("-- Grade Summary --")
  print("Quizzes (20%): {:.1f}%".format(quiz_grade * 100))
  print("Assign1 (15%): {:.1f}%".format(ass1_grade * 100))
  print("Assign2 (15%): {:.1f}%".format(ass2_grade * 100))
  print("Assign3 (20%): {:.1f}%".format(ass3_grade * 100))
  print("Project (30%): {:.1f}%".format(project_grade * 100))
  grade_score = ((quiz_grade * 20 + ass1_grade * 15 + ass2_grade * 15 + ass3_grade * 20 + project_grade * 30))

  letter_grade = threshold_grade(grade_score)
  andrews[andrew_id] = letter_grade
  letters[letter_grade] += 1
  print("Total for {}: {:.1f}% ({})\n".format(name, grade_score, letter_grade))
  print("-- Quiz Details --")
  for qdate, qstr in zip([stats[0][x] for x in quiz_columns], quizzes):
    print("{}: {}".format(qdate, qstr[1])) 

# Example
calc_and_print_grades('Graham Neubig', 'gneubig', [2,3,3,3,3,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3], 96, 92, 96, 100)


def mod_quiz(qid, qstr):
  m = miss_re.match(qstr)
  if qstr == '' or m:
    return (0.0, '0 / {} (Missed)'.format(denoms[qid]))
  m = ex_re.search(qstr)
  if m:
    return (1000.0, qstr)
  m = quiz_re.match(qstr)
  if m:
    return (float(m.group(1))/denoms[qid], f'{m.group(1)} / {denoms[qid]}')
  m = quiz_re2.match(qstr)
  if m:
    return (float(m.group(1))/denoms[qid], f'{m.group(1)} / {denoms[qid]}')
  raise RuntimeError("bad string {}".format(qstr))


with open('2021-12-17T0003_Grades-11711.csv', 'r') as tsvin:
  stats = list(csv.reader(tsvin))
  for row in stats[2:]:

    print("\n\n------------------ {} -------------------".format(row[3]))
    print("\nHi {},\n".format(row[0].split(', ')[1]))
    print("Here is your grade report for CS11-711 including your quizzes and assignments. The individual grades in this summary should match your scores in Canvas. The weighting to obtain your final grade follows the scheme described at the beginning of class and on the class web site. If there are any inaccuracies in this report, please email the TA mailing list as soon as possible.\n\n Thanks a lot for taking the class, and I hope it's been useful to you!\n\nGraham")

    # Get quiz and assignment grades
    quizzes = [mod_quiz(i, row[idx]) for i, idx in enumerate(quiz_columns)]
    ass1_grade, ass2_grade, ass3_grade, project_grade = [(float(row[x]) if row[x] else 0.0)/100.0 for x in assign_columns]

    calc_and_print_grades(row[0], row[3], quizzes, ass1_grade, ass2_grade, ass3_grade, project_grade)

    print("\n***** End {} *****".format(row[0]))
    
for k, v in sorted(letters.items()):
  print(k,v)


calc_and_print_grades()