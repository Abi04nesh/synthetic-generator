import os
import pandas as pd
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from collections import Counter
from django.conf import settings

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

            # Handle missing values based on column type
            for col in df.columns:
                if df[col].isnull().sum() > 0:  # Only handle missing values if any
                    if df[col].dtype in ["int64", "float64"]:
                        # Impute numerical values with the mean
                        df[col] = df[col].fillna(df[col].mean())
                    elif df[col].dtype == "object":
                        # Impute string columns intelligently
                        most_frequent = impute_strings_by_category(df, col)
                        df[col] = df[col].fillna(most_frequent)
                    elif df[col].dtype in ["datetime64[ns]", "datetime64"]:
                        # Impute dates with forward fill
                        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

            # Save the cleaned dataset for download
            cleaned_file_path = os.path.join(settings.MEDIA_ROOT, "cleaned_dataset.csv")
            df.to_csv(cleaned_file_path, index=False)

            # Provide download URL and success message
            context = {
                "message": "Dataset processed successfully.",
                "download_url": fs.url("cleaned_dataset.csv"),
            }

        except Exception as e:
            context["error"] = f"Error processing file: {str(e)}"

    return render(request, "analyzer/upload.html", context)


def impute_strings_by_category(df, col):
    """
    Impute missing string values intelligently based on other columns in the dataset.
    """
    # Get all rows where the column is not null
    non_null_rows = df[df[col].notnull()]

    # Identify categories from another column (e.g., assuming 'Category' column exists)
    if "Category" in df.columns:
        category_map = non_null_rows.groupby("Category")[col].apply(lambda x: Counter(x).most_common(1)[0][0]).to_dict()
        return df["Category"].map(category_map).fillna(df[col].mode()[0])  # Use most frequent as fallback
    else:
        # Fallback: Use the most frequent value in the column
        return df[col].mode()[0]
