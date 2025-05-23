# AI Loan Default Predictor App

An end-to-end machine learning application that predicts loan default risk using **fully synthetic data**.  
Built with a Flask API serving a LightGBM model, and an Angular frontend for interactive input and real-time prediction.

**Live Demo**: [ai.fullstackista.com/ai-loan-default-predictor](https://ai.fullstackista.com/ai-loan-default-predictor)

---

## Project Overview

This app simulates a real-world loan default prediction tool, allowing users to:

- Input realistic applicant details (synthetically generated or manual entry)
- Trigger a pre-trained ML model
- View a predicted default probability score

The backend model is trained entirely on **synthetic data generated via a custom Python script** (see below).  
No real-world or proprietary datasets are used.

---

## Tech Stack

| Layer       | Tech                    |
| ----------- | ----------------------- |
| Frontend    | Angular (HTML, CSS, TS) |
| Backend API | Flask (Python)          |
| ML Model    | LightGBM                |
| Hosting     | AWS EC2 + Docker        |

---

## Key Features

- Interactive UI for simulating loan applications
- One-click test data generation based on feature distributions
- Real-time default risk prediction using a LightGBM model
- Clean architecture with decoupled frontend/backend
- Fully containerized via Docker

---

## Synthetic Data Generation

All data used in this app is generated from scratch using randomization and simulation logic.  
Feature distributions are inspired by public credit datasets, but no actual data is used.

🔗 [View synthetic_data_generator.py](backend-flask/tools/synthetic_data_generator.py)

---

## Live Demo

Try the app here:  
👉 [ai.fullstackista.com/ai-loan-default-predictor](https://ai.fullstackista.com/ai-loan-default-predictor)

**How to use:**

1. Click "Generate Test Data" or enter custom values.
2. Click "Predict Default".
3. View the risk probability score.

No login required.

---

## Future Improvements

- Add SHAP-based model explainability for transparency
- Enable saving and comparing multiple predictions

---

## About the Author

Built by a C-level banking executive with 20+ years in finance, now leading full-stack AI solutions from prototype to deployment.

[Read more at fullstackista.com → About](https://www.fullstackista.com/about-us)

---

## Disclaimer

All data used in this application is fully synthetic.

Feature names follow a realistic structure commonly seen in credit scoring datasets,  
but all values are generated from scratch using custom simulation scripts.  
No real-world, proprietary, or competition datasets are used or referenced.

This project is intended solely for educational and portfolio demonstration purposes.

---

## License

This project is licensed under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).  
You may view, learn from, and adapt the code for non-commercial purposes, with attribution.
