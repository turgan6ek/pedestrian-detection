import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

class RLAgent:
    def __init__(self, state_size=(10, 10), learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.q_table = np.zeros(state_size)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def reward_function(self, detections, ground_truth, iou_threshold=0.5):
        """
        Calculates reward based on detections compared to ground truth.
        """
        true_positives = 0
        false_positives = 0
        false_negatives = len(ground_truth)

        for det in detections:
            matched = False
            for gt in ground_truth:
                if self.calculate_iou(det, gt) > iou_threshold:
                    true_positives += 1
                    false_negatives -= 1
                    matched = True
                    break
            if not matched:
                false_positives += 1

        reward = (10 * true_positives) - (5 * false_positives) - (20 * false_negatives)
        return reward

    def calculate_iou(self, box1, box2):
        """
        Calculates IoU between two bounding boxes.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(10)), random.choice(range(10))
        return np.unravel_index(np.argmax(self.q_table[state]), self.q_table.shape)

    def update(self, state, action, reward, next_state):
        self.q_table[state] += self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[state]
        )
        self.epsilon *= self.epsilon_decay

    def plot_q_table(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.q_table, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Q-table Heatmap")
        plt.xlabel("IoU Threshold")
        plt.ylabel("Confidence Threshold")
        plt.show()
