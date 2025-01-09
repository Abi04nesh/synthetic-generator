import pandas as pd
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from sklearn.impute import SimpleImputer
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

        # Fill numerical columns using RandomForestRegressor
        for column in cleaned_df.select_dtypes(include=["float64", "int64"]):
            if cleaned_df[column].isnull().sum() > 0:
                target = cleaned_df[column]
                predictors = cleaned_df.drop(columns=[column])

                # Handle missing values in predictors
                imputer = SimpleImputer(strategy="mean")
                predictors_imputed = imputer.fit_transform(pd.get_dummies(predictors, drop_first=True))
                
                # Drop rows where target is null
                valid_indices = ~target.isnull()
                X = predictors_imputed[valid_indices]
                y = target[valid_indices]

                if len(y) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = RandomForestRegressor(random_state=42)
                    model.fit(X_train, y_train)

                    # Predict missing values
                    missing_indices = target.isnull()
                    predictors_missing = predictors_imputed[missing_indices]
                    cleaned_df.loc[missing_indices, column] = model.predict(predictors_missing)

        # Fill categorical columns using RandomForestClassifier
        for column in cleaned_df.select_dtypes(include=["object"]):
            if cleaned_df[column].isnull().sum() > 0:
                target = cleaned_df[column]
                predictors = cleaned_df.drop(columns=[column])

                # Handle missing values in predictors
                imputer = SimpleImputer(strategy="most_frequent")
                predictors_imputed = imputer.fit_transform(pd.get_dummies(predictors, drop_first=True))

                # Drop rows where target is null
                valid_indices = ~target.isnull()
                X = predictors_imputed[valid_indices]
                y = target[valid_indices]

                if len(y) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)

                    # Predict missing values
                    missing_indices = target.isnull()
                    predictors_missing = predictors_imputed[missing_indices]
                    cleaned_df.loc[missing_indices, column] = model.predict(predictors_missing)

        # Save cleaned dataset
        cleaned_file_path = os.path.join(settings.MEDIA_ROOT, "cleaned_" + dataset_file.name)
        cleaned_df.to_csv(cleaned_file_path, index=False)

        # Provide cleaned file for download
        context["download_url"] = settings.MEDIA_URL + "cleaned_" + dataset_file.name

    return render(request, "analyzer/upload.html", context)
