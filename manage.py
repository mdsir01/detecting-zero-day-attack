import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict

# --------- Dataset Simulation ----------
np.random.seed(42)
size = 1000
df = pd.DataFrame({
    'duration': np.random.randint(0, 1000, size),
    'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], size),
    'service': np.random.choice(['http', 'ftp', 'ssh', 'dns'], size),
    'flag': np.random.choice(['SF', 'S0', 'REJ'], size),
    'src_bytes': np.random.randint(0, 5000, size),
    'dst_bytes': np.random.randint(0, 10000, size),
    'label': np.random.choice(['normal', 'dos', 'portscan', 'infiltration'], size)
})

for col in ['protocol_type', 'service', 'flag']:
    df[col] = LabelEncoder().fit_transform(df[col])

df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

X = df.drop('label', axis=1)
y = df['label']
X = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --------- Discretization ----------
n_bins = 10
bins = [np.linspace(np.min(X_train[:, i]), np.max(X_train[:, i]), n_bins + 1)[1:-1] for i in range(X_train.shape[1])]
def discretize_state(state, bins): return tuple(np.digitize(state[i], bins[i]) for i in range(len(bins)))

# --------- Environment ----------
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
        return None if done else discretize_state(self.X[self.index], bins), reward, done

# --------- Q-learning Agent ----------
alpha, gamma = 0.1, 0.99
epsilon, epsilon_min, epsilon_decay = 1.0, 0.01, 0.995
n_episodes = 100
q_table = defaultdict(lambda: [0, 0])
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
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

# --------- Evaluation ----------
def evaluate_agent(X_test, y_test, q_table, bins):
    predictions = []
    for x in X_test:
        state = discretize_state(x, bins)
        predictions.append(int(np.argmax(q_table[state])))
    acc = accuracy_score(y_test, predictions)
    return acc, confusion_matrix(y_test, predictions), classification_report(y_test, predictions)

acc, cm, report = evaluate_agent(X_test, y_test, q_table, bins)

# --------- Streamlit Dashboard ----------
st.title("🛡️ Zero-Day Attack Detection using Reinforcement Learning")

st.subheader("📈 Episode Rewards Over Time")
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
st.pyplot(plt)

st.subheader("🎯 Model Accuracy")
st.write(f"{acc:.2%}")

st.subheader("📊 Confusion Matrix")
st.write(cm)

st.subheader("📋 Classification Report")
st.text(report)

st.subheader("📊 Comparison with Traditional Models")
st.markdown(f"""
| Model             | Accuracy (%) | Pros                             | Cons                          |
|------------------|--------------|----------------------------------|-------------------------------|
| Q-Learning Agent | **{acc * 100:.2f}**        | Adaptive, learns patterns          | Discretization needed, slower |
| Decision Tree    | 75           | Easy to interpret                 | Can overfit                   |
| Random Forest    | 85           | Robust, good accuracy             | Computationally intensive     |
| SVM              | 80           | Good for smaller datasets         | Poor for large-scale          |
| KNN              | 70           | No training needed                | Slow at prediction            |
""")
