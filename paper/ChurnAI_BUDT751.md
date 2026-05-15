# ChurnAI: Predicting Telecom Customer Churn from 12 Months of Behavior with an Attention-Based BiLSTM

**Brandon · Chaitra · Hillary · Isabelle · Venkata · Yash**
*BUDT 751 — Harnessing AI for Business · Robert H. Smith School of Business, University of Maryland*

> **Abstract.** Most telecom churn models read a single snapshot of who a customer is today. We think the harder and more useful question is what their behavior has been doing for the last twelve months. ChurnAI is a dashboard-and-chatbot system built around a bidirectional LSTM with attention, trained to spot the kind of slow-then-sharp decline in data usage, logins, and call minutes that usually precedes a cancellation. On a held-out test set of 300 customers the model lands at AUC = 0.95, accuracy = 0.89, recall = 0.84, precision = 0.78, and F1 = 0.80. We also show that the attention weights are themselves a useful diagnostic: they tell an analyst not only that a customer is at risk, but which month of the customer's history the model is actually reacting to.

(See `ChurnAI_BUDT751.docx` for the formatted version with tables and embedded figures.)
