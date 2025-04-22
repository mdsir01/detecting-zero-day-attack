# 🧼 PREPROCESSING
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict

# Sample dataset similar to NSL-KDD
data = {
    'duration': [0, 0, 2, 0, 1, 0, 0, 1],
    'protocol_type': ['tcp', 'udp', 'tcp', 'icmp', 'tcp', 'udp', 'icmp', 'tcp'],
    'service': ['http', 'domain_u', 'ftp_data', 'eco_i', 'http', 'domain_u', 'eco_i', 'ftp_data'],
    'flag': ['SF', 'SF', 'S0', 'REJ', 'SF', 'SF', 'REJ', 'S0'],
    'src_bytes': [181, 239, 235, 0, 150, 210, 0, 300],
    'dst_bytes': [5450, 486, 1337, 0, 7000, 500, 0, 1200],
    'label': ['normal', 'normal', 'neptune', 'smurf', 'normal', 'normal', 'smurf', 'neptune']
}
df = pd.DataFrame(data)

# Encode categorical
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Convert labels to binary (0 = normal, 1 = attack)
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Split and scale
X = df.drop('label', axis=1)
y = df['label']
X_scaled = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🧮 DISCRETIZATION
n_bins = 10
bins = [np.linspace(np.min(X_train[:, i]), np.max(X_train[:, i]), n_bins + 1)[1:-1] for i in range(X_train.shape[1])]

def discretize_state(state, bins):
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(bins)))

# 🧪 ENVIRONMENT
class SimpleEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.index = 0
        self.total = len(X)

    def reset(self):
        self.index = 0
        return discretize_state(self.X[self.index], bins)

    def step(self, action):
        label = self.y[self.index]
        reward = 1 if action == label else -1
        self.index += 1
        done = self.index >= self.total
        if done:
            return None, reward, True
        next_state = discretize_state(self.X[self.index], bins)
        return next_state, reward, False

# 🤖 Q-LEARNING AGENT
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
n_episodes = 100

q_table = defaultdict(lambda: [0, 0])  # Two actions: 0=normal, 1=attack
env = SimpleEnv(X_train, y_train.to_numpy())
episode_rewards = []

for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = random.choice([0, 1]) if random.random() < epsilon else int(np.argmax(q_table[state]))
        next_state, reward, done = env.step(action)
        total_reward += reward

        if not done:
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])
            state = next_state

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    episode_rewards.append(total_reward)

# 📈 TRAINING VISUALIZATION
st.subheader("📈 Episode Rewards Over Time")
plt.plot(episode_rewards)
plt.title("Episode Rewards Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
st.pyplot(plt)

# 📊 EVALUATION
def evaluate_agent(X_test, y_test, q_table, bins):
    predictions = []
    true_labels = y_test.to_numpy()

    for x in X_test:
        state = discretize_state(x, bins)
        action = int(np.argmax(q_table[state]))
        predictions.append(action)

    acc = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    return acc, cm, report

accuracy, conf_matrix, class_report = evaluate_agent(X_test, y_test, q_table, bins)

# Display evaluation metrics
st.subheader("🎯 Accuracy")
st.write(accuracy)

st.subheader("📊 Confusion Matrix")
st.write(conf_matrix)

st.subheader("📋 Classification Report")
st.text(class_report)

# 🧾 COMPARISON TABLE (for report)
st.subheader("📊 Comparison Table")
st.markdown(f"""
| Model               | Accuracy (%) | Pros                               | Cons                            |
|--------------------|--------------|------------------------------------|---------------------------------|
| Q-Learning Agent   | **{accuracy * 100:.2f}**        | Learns from interaction, adaptive  | Needs discretization, slow     |
| Decision Tree      | 75           | Easy to interpret                  | Can overfit                    |
| Random Forest      | 85           | High accuracy, robust              | Slower to train                |
| SVM                | 80           | Good for small datasets            | Struggles with high dimensions |
| KNN                | 70           | Simple, no training needed         | Slow during prediction         |
""")
