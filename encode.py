import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score


judge1_predictions = [
    "Imagist Poem",
    "Nursery Rhyme",
    "Children's Poetry",
    "Children's Poetry",
    "Philosophical Poem",
    "Reflection Poetry",
    "Lyric Poem",
    "Freeverse Poetry",
    "Modernist Poetry",
    "Moral/Didactic Poetry",
    "Narrative Poetry",
    "Gothic Poetry",
    "Inspirational Poetry",
    "Inspirational Poetry",
    "African-American Poetry",
    "Inspirational Poetry",
    "Nature Poetry",
    "Romance Poetry",
    "Reflection Poetry",
    "Reflection Poetry",
    "Reflection Poetry",
    "Reflection Poetry",
    "Confessional Poetry",
    "Modernist Poetry",
    "Reflection Poetry",
    "Reflection Poetry",
    "Inspirational Poetry",
    "Romance Poetry",
    "Mystical Poetry",
    "Philosophical Poem",
    "Contemporary Poetry",
    "Descriptive Poetry",
    "Descriptive Poetry",
    "Lyric Poem",
    "Contemporary Poetry",
    "Romance Poetry",
    "Inspirational Poetry",
    "Inspirational Poetry",
    "Freeverse Poetry"
]


judge2_predictions = [
    "Descriptive Poetry",
    "Nursery Rhyme",
    "Children's Poetry",
    "Children's Poetry",
    "Reflection Poetry",
    "Narrative Poetry",
    "Lyric Poem",
    "Modernist Poetry",
    "Imagist Poem",
    "Moral/Didactic Poetry",
    "Descriptive Poetry",
    "Gothic Poetry",
    "Inspirational Poetry",
    "Moral/Didactic Poetry",
    "African-American Poetry",
    "African-American Poetry",
    "Descriptive Poetry",
    "Lyric Poem",
    "Freeverse Poetry",
    "Freeverse Poetry",
    "Freeverse Poetry",
    "Lament Poetry",
    "Dramatic Monologue",
    "Introspective Poetry",
    "Confessional Poetry",
    "Reflection Poetry",
    "Inspirational Poetry",
    "Romance Poetry",
    "Mystical Poetry",
    "Philosophical Poem",
    "Mystical Poem",
    "Descriptive Poetry",
    "Freeverse Poetry",
    "Romance Poetry",
    "Reflection Poetry",
    "Inspirational Poetry",
    "Descriptive Poetry",
    "Inspirational Poetry",
    "Narrative Poetry"
]

judge3_predictions = [
    "Introspective Poetry",
    "Children's Poetry",
    "Children's Poetry",
    "Narrative Poetry",
    "Philosophical Poem",
    "Reflection Poetry",
    "Descriptive Poetry",
    "Imagist Poem",
    "Imagist Poem",
    "Gothic Poetry",
    "Philosophical Poem",
    "Gothic Poetry",
    "Reflection Poetry",
    "Dramatic Monologue",
    "Contemporary Poetry",
    "Inspirational Poetry",
    "Descriptive Poetry",
    "Romance Poetry",
    "Narrative Poetry",
    "Freeverse Poetry",
    "Confessional Poetry",
    "Dramatic Monologue",
    "Confessional Poetry",
    "Romance Poetry",
    "Dramatic Monologue",
    "Inspirational Poetry",
    "Inspirational Poetry",
    "Romance Poetry",
    "Freeverse Poetry",
    "Contemporary Poetry",
    "Freeverse Poetry",
    "Philosophical Poem",
    "Introspective Poetry",
    "Introspective Poetry",
    "Reflection Poetry",
    "Freeverse Poetry",
    "Lyric Poem",
    "Freeverse Poetry",
    "Lyric Poem"
]

judge4_predictions = [
    "Imagist Poem",
    "Nursery Rhyme",
    "Narrative Poetry",
    "Children's Poetry",
    "Philosophical Poem",
    "Narrative Poetry",
    "Lyric Poem",
    "Imagist Poem",
    "Dramatic Monologue",
    "Lyric Poem",
    "Narrative Poetry",
    "Narrative Poetry",
    "Philosophical Poetry",
    "Inspirational Poetry",
    "Inspirational Poetry",
    "Inspirational Poetry",
    "Romance Poetry",
    "Romance Poetry",
    "Narrative Poetry",
    "Dramatic Monologue",
    "Inspirational Poetry",
    "Dramatic Monologue",
    "Contemporary Poetry",
    "Contemporary Poetry",
    "Contemporary Poetry",
    "Inspirational Poetry",
    "Inspirational Poetry",
    "Romance Poetry",
    "Introspective Poetry",
    "Introspective Poetry",
    "Descriptive Poetry",
    "Descriptive Poetry",
    "Introspective Poetry",
    "Lyric Poem",
    "Imagist Poem",
    "Inspirational Poetry",
    "Lyric Poem",
    "Reflection Poetry",
    "Autobiographical Poetry"
]


judge5_predictions = [
    "Lyric Poem",
    "Nursery Rhyme",
    "Children's Poetry",
    "Children's Poetry",
    "Motivational Poem",
    "Philosophical Poem",
    "Lyric Poem",
    "Imagist Poem",
    "Dramatic Monologue",
    "Lyric Poem",
    "Lyric Poem",
    "Gothic Poetry",
    "Lyric Poem",
    "Moral/Didactic Poetry",
    "African-American Poetry",
    "Autobiographical Poetry",
    "Romance Poetry",
    "Romance Poetry",
    "Narrative Poetry",
    "Freeverse Poetry",
    "Freeverse Poetry",
    "Freeverse Poetry",
    "Confessional Poetry",
    "Contemporary Poetry",
    "Confessional Poetry",
    "Poetry",
    "Inspirational Poetry",
    "Romance Poetry",
    "Reflective Poetry",
    "Philosophical Poem",
    "Lyrical Poetry",
    "Reflective Poetry",
    "Introspective Poetry",
    "Mystical Poetry",
    "Nature Poetry",
    "Nature Poetry",
    "Descriptive Poetry",
    "Lyric Poem",
    "Narrative Poetry",
]


"""all_predictions = judge1_predictions + judge2_predictions + judge3_predictions + judge4_predictions

# Get distinct categories
distinct_categories = set(all_predictions)
print(distinct_categories)
# Count the number of distinct categories
num_categories = len(distinct_categories)

# Print the result
print("Number of distinct categories:", num_categories)
"""

all_predictions = [judge1_predictions, judge2_predictions,
                   judge3_predictions, judge4_predictions, judge5_predictions]

# Get the majority vote for each instance
majority_vote_predictions = [
    "Imagist Poem",
    "Nursery Rhyme",
    "Children's Poetry",
    "Children's Poetry",
    "Philosophical Poem",
    "Reflection Poetry",
    "Lyric Poem",
    "Imagist Poem",
    "Imagist Poem",
    "Moral/Didactic Poetry",
    "Narrative Poetry",
    "Gothic Poetry",
    "Inspirational Poetry",
    "Inspirational Poetry",
    "African-American Poetry",
    "Inspirational Poetry",
    "Descriptive Poetry",
    "Romance Poetry",
    "Narrative Poetry",
    "Freeverse Poetry",
    "Confessional Poetry",
    "Dramatic Monologue",
    "Confessional Poetry",
    "Introspective Poetry",
    "Confessional Poetry",
    "Reflection Poetry",
    "Inspirational Poetry",
    "Romance Poetry",
    "Mystical Poetry",
    "Philosophical Poem",
    "Descriptive Poem",
    "Descriptive Poetry",
    "Introspective Poetry",
    "Lyric Poem",
    "Reflection Poetry",
    "Inspirational Poetry",
    "Lyric Poem",
    "Inspirational Poetry",
    "Narrative Poetry"
]
"""
for i in range(len(all_predictions[0])):
    instance_predictions = [pred[i] for pred in all_predictions]
    majority_vote = max(set(instance_predictions), key=instance_predictions.count)
    majority_vote_predictions.append(majority_vote)
"""


"""l = [len(judge1_predictions), len(judge2_predictions), len(judge3_predictions), len(judge4_predictions), len(judge5_predictions), len(majority_vote_predictions)]
print(l)
"""


# print(majority_vote_predictions)
# Calculate the F1 score for each individual judge's predictions

f1_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []


for judge_predictions in all_predictions:
    f1 = f1_score(majority_vote_predictions,
                  judge_predictions, average='weighted')
    accuracy = accuracy_score(
        majority_vote_predictions, judge_predictions)
    precision = precision_score(
        majority_vote_predictions, judge_predictions, average='weighted')
    recall = recall_score(majority_vote_predictions,
                          judge_predictions, average='weighted')

    f1_scores.append(f1)
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)


print("F1 Scores:", f1_scores)
print("Accuracy Scores:", accuracy_scores)
print("Precision Scores:", precision_scores)
print("Recall Scores:", recall_scores)


judge_names = ['Judge 1', 'Judge 2', 'Judge 3', 'Judge 4', 'Chat-GPT']

# Set the width of the bars
bar_width = 0.2

# Set the positions of the bars on the x-axis
bar_positions = np.arange(len(judge_names))

# Plotting the scores
plt.figure(figsize=(12, 8))
plt.bar(bar_positions - bar_width * 1.5, f1_scores,
        width=bar_width, label='F1 Score')
plt.bar(bar_positions - bar_width * 0.5, accuracy_scores,
        width=bar_width, label='Accuracy Score')
plt.bar(bar_positions + bar_width * 0.5, precision_scores,
        width=bar_width, label='Precision Score')
plt.bar(bar_positions + bar_width * 1.5, recall_scores,
        width=bar_width, label='Recall Score')

# Set the x-axis tick labels
plt.xticks(bar_positions, judge_names)

# Set the labels and title
plt.xlabel('Judge')
plt.ylabel('Scores')
plt.title('Performance Scores by Judge')

# Add a legend
plt.legend()

# Display the plot
plt.show()


cm = confusion_matrix(majority_vote_predictions, judge5_predictions)

# Getting unique class labels from both arrays
class_labels = np.unique(np.concatenate(
    (majority_vote_predictions, judge5_predictions)))

# Extracting the first word from each label
class_labels = [label.split()[0] for label in class_labels]

# Adjusting figure size based on the number of unique class labels
fig, ax = plt.subplots(figsize=(len(class_labels)+2, len(class_labels)+2))

# Shifting the heatmap upward
cax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                  xticklabels=class_labels, yticklabels=class_labels, ax=ax)
cax.set_ylim(len(class_labels), 0)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')


plt.show()
