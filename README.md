# Zepto Data & AI Project

This repository contains three modules developed as part of the Zepto technical project.

## Project Structure

```text
module1/
│
├── data_pipeline/
│   ├── Data Pipeline.py
│   ├── books_catalog.db
│   ├── books_cleaned.csv
│   └── sql_query_outputs.txt
│
├── analytics/
│   ├── analytics_pipeline.py
│   ├── titanic.csv
│   ├── best_titanic_pipeline.joblib
│   ├── classifier_comparison.csv
│   ├── final_model_comparison.csv
│   ├── imbalance_comparison.csv
│   ├── charts/
│   │   ├── confusion_matrices.png
│   │   ├── decision_tree.png
│   │   ├── fare_residual_plot.png
│   │   └── roc_curves.png
│   └── README.md
│
├── support_assistant/
│   ├── main.py
│   ├── docs/
│   │   ├── doc_1.txt
│   │   ├── doc_2.txt
│   │   ├── doc_3.txt
│   │   ├── doc_4.txt
│   │   ├── doc_5.txt
│   │   ├── doc_6.txt
│   │   ├── doc_7.txt
│   │   └── doc_8.txt
│   ├── chroma_db/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
└── README.md
Module 1 — Data Pipeline

The Data Pipeline module collects book information from the web, cleans and processes the collected data, stores the catalog in a SQLite database, and generates SQL query outputs.

Main Features
Web data collection
Data cleaning
Data transformation
SQLite database storage
SQL queries and outputs
CSV data generation
Main Files
Data Pipeline.py — main data pipeline
books_cleaned.csv — cleaned book dataset
books_catalog.db — SQLite database
sql_query_outputs.txt — SQL query results
How to Run

Open PowerShell in the project root and run:

cd data_pipeline
python "Data Pipeline.py"
Module 2 — Analytics Pipeline

The Analytics Pipeline module performs exploratory data analysis and predictive modeling using the Titanic dataset.

Main Features
Dataset loading
Exploratory Data Analysis (EDA)
Data cleaning
Feature preprocessing
Handling class imbalance
Machine learning model training
Model comparison
Model evaluation
Data visualization
Main Files
analytics_pipeline.py — main analytics and modeling pipeline
titanic.csv — Titanic dataset
best_titanic_pipeline.joblib — saved trained pipeline
classifier_comparison.csv — classifier comparison results
final_model_comparison.csv — final model comparison
imbalance_comparison.csv — imbalance handling comparison
charts/ — generated visualizations
How to Run

Open PowerShell in the project root and run:

cd analytics
python analytics_pipeline.py
Module 3 — Support Assistant

The Support Assistant module is a Retrieval-Augmented Generation (RAG) based customer-support assistant for answering questions using a local knowledge base.

Main Features
Local support knowledge base
Document loading
Text embeddings
Semantic document retrieval
ChromaDB vector database
Support question answering
Main Files
main.py — main support assistant application
docs/ — support policy documents
chroma_db/ — vector database
requirements.txt — Python dependencies
Dockerfile — Docker configuration
How to Run

Open PowerShell in the project root and run:

cd support_assistant
pip install -r requirements.txt
python main.py
Setup
Requirements
Python 3.12 or compatible Python version
Git
Internet connection for required package installation and Module 1 data collection
Install Module 3 Dependencies
cd support_assistant
pip install -r requirements.txt
Design Decisions
Module 1 — Data Pipeline

The pipeline separates data collection, cleaning, database storage, and SQL analysis. This makes each stage easier to understand, test, and maintain.

Module 2 — Analytics Pipeline

The analytics workflow combines data exploration, preprocessing, imbalance handling, model training, and evaluation. Machine learning preprocessing is organized into a pipeline to provide a consistent modeling workflow.

Module 3 — Support Assistant

The support assistant uses a local document knowledge base and vector similarity search. ChromaDB is used to store document embeddings and retrieve relevant support information for user questions.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
SQLite
ChromaDB
Sentence Transformers
Git
GitHub
Docker
End-to-End Execution

The three modules can be executed independently from their respective folders.

Module 1
cd data_pipeline
python "Data Pipeline.py"
Module 2
cd analytics
python analytics_pipeline.py
Module 3
cd support_assistant
pip install -r requirements.txt
python main.py
Repository Submission

This is a single public GitHub repository containing all three required modules:

data_pipeline
analytics
support_assistant

The root README.md provides the project structure, setup instructions, execution commands, and design decisions for each module.

