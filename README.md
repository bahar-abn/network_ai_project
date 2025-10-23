# 🧠 Network AI Project

Hi there! 👋  
This is a small project I built to connect **AI** and **network monitoring** together — basically, a machine learning model that analyses captured network traffic and tries to understand if it’s normal or suspicious.

I made this as part of my first month learning roadmap about **AI and networking fundamentals**.  
The goal was to practise things like data preprocessing, PyTorch basics, and simple network traffic analysis.



## 💻 How to Run

### 1️⃣ Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
2️⃣ Install requirements
bash
Copy code
pip install -r requirements.txt
3️⃣ Capture some network traffic (Mac)
bash
Copy code
sudo tcpdump -i en0 -c 50 -w data/traffic_raw.pcap
4️⃣ Extract features
bash
Copy code
python src/preprocess.py
5️⃣ Train the model
bash
Copy code
python -m src.train
6️⃣ Evaluate results
bash
Copy code
python -m src.evaluate
📊 Example Output
Training:

yaml
Copy code
Epoch 1, Loss: 0.77
Epoch 5, Loss: 0.56
Model trained and saved!
Evaluation:

lua
Copy code
Accuracy: 0.82
Confusion Matrix:
[[41  0]
 [ 9  0]]

---
🧩 What I Learned
How to collect and read real network packets with tcpdump and scapy.

Converting raw data into features for machine learning.

Building and training a simple PyTorch model.

Evaluating accuracy and confusion matrix.

Connecting basic AI with cybersecurity ideas.

---

🚀 Future Plans
I’d love to:

Add more traffic samples and train deeper models.

Try anomaly detection with autoencoders.

Visualise the data better with matplotlib or seaborn.

