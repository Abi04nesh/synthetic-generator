import pandas as pd
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
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

        # Analyze and clean dataset
        cleaned_df = df.copy()

        # Fill numerical missing values using interpolation
        for column in cleaned_df.select_dtypes(include=["float64", "int64"]):
            if cleaned_df[column].isnull().sum() > 0:
                cleaned_df[column] = cleaned_df[column].interpolate(method='linear', limit_direction='forward', axis=0)
                # Fill remaining NaN (if any after interpolation) with mean
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)

        # Fill string missing values based on category frequencies
        for column in cleaned_df.select_dtypes(include=["object"]):
            if cleaned_df[column].isnull().sum() > 0:
                mode_value = cleaned_df[column].mode()[0]
                cleaned_df[column].fillna(mode_value, inplace=True)

        # Save cleaned dataset
        cleaned_file_path = os.path.join(settings.MEDIA_ROOT, "cleaned_" + dataset_file.name)
        cleaned_df.to_csv(cleaned_file_path, index=False)

        # Provide cleaned file for download
        context["download_url"] = settings.MEDIA_URL + "cleaned_" + dataset_file.name

    return render(request, "analyzer/upload.html", context)
