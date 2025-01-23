# abi04nesh Synthetic Generator

This project, **abi04nesh Synthetic Generator**, is a Django-based application designed for dataset processing and analysis. The project includes tools for uploading datasets, performing analysis, handling missing values, and generating cleaned datasets.



## Features
- **Dataset Upload**: Users can upload CSV datasets via the web interface.
- **Data Cleaning**: Missing values are handled using advanced techniques such as `RandomForestRegressor` and `RandomForestClassifier`.
- **Cleaned Dataset**: A cleaned version of the dataset is generated and available for download.
- **Custom Filters**: Includes reusable custom filters for templates.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd abi04nesh-synthetic-generator
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Apply migrations:
   ```bash
   python manage.py migrate
   ```
5. Run the development server:
   ```bash
   python manage.py runserver
   ```
6. Access the application at `http://127.0.0.1:8000/`.

## Usage
1. Upload a CSV dataset via the upload interface (`upload.html`).
2. The system processes the dataset and handles missing values.
3. Download the cleaned dataset from the provided link.

## Notes
- All temporary datasets and cleaned datasets are stored in the `media/` directory.
- Example datasets are provided for testing.
- Database migrations are stored in the `migrations/` directory of the `analyzer/` app.

