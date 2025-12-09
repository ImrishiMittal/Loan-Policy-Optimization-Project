# Loan Policy Optimization using Deep Learning & Offline Reinforcement Learning

This project focuses on optimizing loan approval decisions using a combination of **Deep Learning** and **Offline Reinforcement Learning (Offline RL)**.  
Using real-world Lending Club loan data, we aim to:

- Predict loan default risk  
- Learn a profit-maximizing loan approval policy  
- Compare Deep Learning and RL-based decision-making approaches  

---

## 📂 Project Structure

Loan-Policy-Optimization-Project/
│
├── EDA/
│ └── eda.ipynb
│
├── DL_MODEL/
│ └── dl_model.ipynb
│
├── RL_MODEL/
│ └── rl_agent.ipynb
│
└── Report/
└── Report.md



---

## 📊 1. Exploratory Data Analysis (EDA)

- Loaded and processed 1.26M+ Lending Club loan records  
- Cleaned missing values  
- Selected relevant financial variables  
- Converted categorical fields using one-hot encoding  
- Converted interest rates (“13.5%”) to numeric  
- Filtered final outcomes:
  - Fully Paid → 0  
  - Charged Off → 1  

The cleaned dataset is saved as:


---

## 🤖 2. Deep Learning Model

A Multi-Layer Perceptron classifier was trained to predict borrower default.

### **Model Architecture**
- Dense(64, ReLU)  
- Dense(32, ReLU)  
- Dense(1, Sigmoid)  

### **Results:**
| Metric | Value |
|-------|--------|
| **AUC** | **0.708** |
| **F1-score** | **0.099** |

### Interpretation
- The model successfully separates good vs risky borrowers (AUC 0.708).  
- F1-score is low due to dataset imbalance (80% fully paid, 20% default).  
- Deep Learning is good for **risk prediction**, not for policy optimization.

---

## 🧠 3. Offline Reinforcement Learning

### **RL Problem Setup**

- **State (s):** borrower features  
- **Action (a):**  
  - 1 → Approve  
  - 0 → Deny  
- **Reward (r):**  
  - Approve & Fully Paid → `loan_amount × (interest_rate / 100)`  
  - Approve & Default → `−loan_amount`  
  - Deny → `0`  

### **Method Used**

Since the Colab environment did not support stable RL libraries, we implemented an **Offline Q-Learning regression approach**:

- Construct both actions (`a = 0` and `a = 1`) for each state  
- Train a **RandomForestRegressor** to estimate Q(s, a)  
- Learned policy:


Approve if Q(s,1) > Q(s,0)
Otherwise Deny


### **RL Policy Output Example**


[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]


### **Estimated Policy Value**


+36.51


### Interpretation
- RL policy is **conservative and profit-oriented**.  
- It approves fewer loans but avoids large losses from defaults.  
- RL achieves **positive expected return**, outperforming DL for decision-making.

---

## 🔥 4. Deep Learning vs. Reinforcement Learning

| Aspect | Deep Learning | Reinforcement Learning |
|--------|----------------|------------------------|
| Objective | Predict default | Maximize financial returns |
| Output | Probability | Action (Approve / Deny) |
| Behavior | Approves many loans | Approves few, avoids risk |
| Strength | Good at classification | Good at economic optimization |
| Weakness | Not profit-aware | Very conservative |

---

## 📝 5. Final Report

A detailed project report is included in:



Report/Report.md


It contains:

- EDA findings  
- Model architectures  
- Performance metrics  
- RL policy formulation  
- Expected value estimation  
- DL vs RL comparison  
- Conclusions  

---

## 👤 Author

**Rishi Mittal**  
Machine Learning Project  
B.Tech  

---

## ⭐ Summary

This project demonstrates how:

- Deep Learning helps assess default risk  
- Offline RL helps make profit-optimized decisions  
- Combining both approaches leads to smarter loan policies  

The RL policy learned from historical data provides a more financially safe decision rule, achieving an **estimated average reward of +36.51 per loan**.
