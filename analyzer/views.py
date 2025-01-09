import pandas as pd
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

def upload_dataset(request):
    context = {}
    if request.method == "POST" and request.FILES.get("dataset"):
        # Save the uploaded file
        dataset_file = request.FILES["dataset"]
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        file_path = fs.save(dataset_file.name, dataset_file)
        file_full_path = fs.path(file_path)

        # Read the dataset
        try:
            df = pd.read_csv(file_full_path)
        except Exception as e:
            context["error"] = f"Error reading the file: {e}"
            return render(request, "analyzer/upload.html", context)

        # Data Cleaning and ML-Based Imputation
        cleaned_df = df.copy()

        # Replace empty strings and invalid values with NaN
        cleaned_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        # Separate numerical and categorical columns
        numeric_columns = cleaned_df.select_dtypes(include=["float64", "int64"]).columns
        categorical_columns = cleaned_df.select_dtypes(include=["object"]).columns

        # -----------------------------
        # Numerical Columns Imputation
        # -----------------------------

        for column in numeric_columns:
            if cleaned_df[column].isnull().sum() > 0:
                target = cleaned_df[column]
                predictors = cleaned_df.drop(columns=[column])

                # Ensure that predictors only contain numeric columns (drop categorical columns)
                predictors = predictors.select_dtypes(include=["float64", "int64"])

                # Use KNNImputer for filling missing values in predictors before applying RandomForestRegressor
                imputer = KNNImputer(n_neighbors=5)  # Using KNN to fill the missing values for predictors
                predictors_imputed = imputer.fit_transform(predictors)

                # Drop rows where target is null
                valid_indices = ~target.isnull()
                X = predictors_imputed[valid_indices]
                y = target[valid_indices]

                if len(y) > 0:
                    # Split into training and testing data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = RandomForestRegressor(random_state=42)
                    model.fit(X_train, y_train)

                    # Predict missing values in target column
                    missing_indices = target.isnull()
                    predictors_missing = predictors_imputed[missing_indices]
                    predictions = model.predict(predictors_missing)

                    # Fill the missing values in the dataframe
                    cleaned_df.loc[missing_indices, column] = predictions

        # -----------------------------
        # Categorical Columns Imputation
        # -----------------------------

        for column in categorical_columns:
            if cleaned_df[column].isnull().sum() > 0:
                target = cleaned_df[column]
                predictors = cleaned_df.drop(columns=[column])

                # Ensure that predictors only contain numeric columns (drop categorical columns)
                predictors = predictors.select_dtypes(include=["float64", "int64"])

                # Use KNNImputer for filling missing values in predictors before applying RandomForestClassifier
                imputer = KNNImputer(n_neighbors=5)  # Using KNN for categorical columns as well
                predictors_imputed = imputer.fit_transform(predictors)

                # Drop rows where target is null
                valid_indices = ~target.isnull()
                X = predictors_imputed[valid_indices]
                y = target[valid_indices]

                if len(y) > 0:
                    # Split into training and testing data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)

                    # Predict missing values in target column
                    missing_indices = target.isnull()
                    predictors_missing = predictors_imputed[missing_indices]
                    predictions = model.predict(predictors_missing)

                    # Fill the missing values in the dataframe
                    cleaned_df.loc[missing_indices, column] = predictions

        # Save cleaned dataset
        cleaned_file_path = os.path.join(settings.MEDIA_ROOT, "cleaned_" + dataset_file.name)
        cleaned_df.to_csv(cleaned_file_path, index=False)

        # Provide cleaned file for download
        context["download_url"] = settings.MEDIA_URL + "cleaned_" + dataset_file.name

    return render(request, "analyzer/upload.html", context)
