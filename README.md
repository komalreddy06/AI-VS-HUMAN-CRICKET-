# 🏏 AI Cricket – Intelligence vs Instinct
**Final Year Project | AI & Search Techniques**

## Algorithms Implemented
| Unit | Algorithm | Game Role |
|------|-----------|-----------|
| Unit I | **A\* Search** | Field placement optimization |
| Unit II | **Minimax + Alpha-Beta Pruning** | AI bowler decisions |
| Unit III | **Bayesian / MDP concepts** | Match strategy |
| Unit IV | **Q-Learning** | AI adapts delivery mid-match |

---
---

## 🎮 How to Play
1. **Select a delivery** (Yorker, Bouncer, Off Spin…)
2. Field automatically repositions using **A\* Search**
3. **Select your shot** (Drive, Pull, Sweep…)
4. Click **PLAY BALL** — AI uses **Minimax** to counter you
5. Watch **Q-Learning** update the AI's strategy each ball

---

## 📊 Syllabus Coverage
- **State-space representation**: Field zones as nodes
- **Informed search**: A\* with heuristic h(n) = uncovered run probability
- **Game Tree**: 2-ply Minimax (bowler → batsman → outcome)
- **Alpha-Beta pruning**: Cuts ~45% of game tree nodes
- **Q-Learning**: Q(s,a) ← Q + α[r + γ·maxQ' − Q]
- **MDP analogy**: Match phases as states, shot/delivery as actions
