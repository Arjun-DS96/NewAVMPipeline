import pandas as pd
import numpy as np
import asyncio
from KNN_faiss import GeoKNNSearch
from property_scraper import PropertyScraper
import re  # Import the scraper class
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

class PropertyAnalyzer:

    def __init__(self, postcode, paon=None, saon=None,mj_valuation_date=None):
        self.postcode = postcode
        self.paon = paon
        self.saon = saon
        self.data_enrich_2025=pd.DataFrame()
        self.data_enrich_train=pd.DataFrame()
        self.sample = self.load_sample()
        self.potential_comparables = []
        self.potential_comparables_df = pd.DataFrame()
        self.dropped_data=pd.DataFrame()
        self.pc_before_ohe = pd.DataFrame()
        self.sample_before_ohe=pd.DataFrame()
        

        # Set default value for mj_valuation_date to today's date if not passed
        if mj_valuation_date is None:
            self.mj_valuation_date = pd.to_datetime(mj_valuation_date, errors='coerce')
        else:
            self.mj_valuation_date = pd.to_datetime(mj_valuation_date, errors='coerce')

    def load_data(self):
        """Loads property dataset and splits into recent (2025) and historical records."""
        data_enrich = pd.read_csv("Data/premium_property_enrich_cleaned.csv")

        data_enrich["Transaction unique identifier"] = (
        data_enrich["Transaction unique identifier"]
        .str.strip()    # Removes leading and trailing spaces
        .str.strip("{}") # Removes the first and last curly braces
        )
        
        data_enrich['Date of Transfer'] = pd.to_datetime(data_enrich['Date of Transfer'])
        start_date, end_date = '2025-01-01', '2025-12-31'
        self.data_enrich_2025 = data_enrich[(data_enrich['Date of Transfer'] >= start_date) &
                                    (data_enrich['Date of Transfer'] <= end_date)]
        self.data_enrich_train = data_enrich[data_enrich['Date of Transfer'] <= start_date]

    
    def load_sample(self):
        self.load_data()
        """Loads a property sample based on postcode, PAON, and SAON."""
        sample = self.data_enrich_train[
            (self.data_enrich_train['Postcode'] == self.postcode)
        ]
        if self.paon:
            sample = sample[sample['PAON (Primary Addressable Object Name)'] == self.paon]
        if self.saon:
            sample = sample[sample['SAON (Secondary Addressable Object Name)'] == self.saon]
        return sample.iloc[0] if not sample.empty else None

    def extract_property_name(self, sample):
        """Constructs property name from SAON, PAON, and Street fields."""
        saon = sample.get("SAON (Secondary Addressable Object Name)", "")
        saon = "" if pd.isna(saon) else saon.strip()
        paon = sample.get("PAON (Primary Addressable Object Name)", "")
        paon = "" if pd.isna(paon) else paon.strip()
        street = sample.get("Street", "").strip()
        street = "" if pd.isna(street) else street.strip()

        if saon:
            return f"{saon}, {paon}, {street}".strip()
        return f"{paon}, {street}".strip()

    def initialize_knn(self):
        """Initializes the KNN model with historical property data."""
        return GeoKNNSearch(
            data=self.data_enrich_train[['Property_Index', 'latitude', 'longitude']],
            lat_col='latitude', lon_col='longitude', id_col='Property_Index',
            use_exact_distance=True
        )

    def scrape_sample_property(self, sample, scraper):
        """Scrapes property details for the sample property and updates it."""
        postcode = sample.get("Postcode", "").strip()
        property_name = self.extract_property_name(sample)

        if postcode and property_name:
            details = scraper.scrape_property_details(postcode, property_name)
        else:
            details = None

        self.update_sample_with_details(sample, details)

    def update_sample_with_details(self, sample, details):
        """Update the sample property with scraped details."""
        if details:
            sample["Total Floor Area"] = details.get("Total Floor Area", "Not Available (Not Scraped)")
            sample["Habitable Rooms"] = details.get("Habitable Rooms", "Not Available (Not Scraped)")
            sample["Heated Rooms"] = details.get("Heated Rooms", "Not Available (Not Scraped)")
            sample["Tenure - Propcheck"] = details.get("Tenure", "Not Available (Not Scraped)")
            sample["Form"] = details.get("Form", "Not Available (Not Scraped)")
            sample["Type"] = details.get("Type", "Not Available (Not Scraped)")
            sample["Year Built"] = details.get("Year Built", "Not Available (Not Scraped)")
            sample["New Build? - Propcheck"] = details.get("New Build", "Not Available (Not Scraped)")
            sample["Current Energy Rating"] = details.get("Current Energy Rating", "Not Available (Not Scraped)")
            estimated_value = details.get("Estimated Value", "Not Available (Not Scraped)")
            sample["Estimated Value"] = re.sub(r"[^\d,]", "", estimated_value)
            sample["Distance"]=0.00
            sample["Days Before MJ Valuation"]=0

    def enrich_potential_comparables(self,potential_comparables, scraper):
        """Scrapes property details for comparable properties and returns a DataFrame with distances."""

        
        property_details_df = pd.DataFrame(columns=[
            'Potential_Comparable_ID', 'Property_Index', 'Distance', 'Date of Transfer', 'Postcode', 'PPD Category',
            'Total Floor Area', 'Habitable Rooms', 'Heated Rooms', 'SAON', 'PAON',
            'Tenure - Gov', 'Tenure - Propcheck', 'New Build? - Gov', 'New Build? - Propcheck',
            'Street', 'Form', 'Type', 'Year Built', 'Current Energy Rating', 'Estimated Value','Days Before MJ Valuation'
        ])
        
        pc_id_counter = 1  # Start ID counter

        for property_index, distance in potential_comparables:  # Now looping over 2D array
            potential_comparable_id = f"PC{pc_id_counter}"
            pc_id_counter += 1  # Increment for next loop

            rows = self.data_enrich_train[self.data_enrich_train['Property_Index'] == property_index]
            
            if not rows.empty:
                # If there are multiple rows, sort by 'Date of Transfer' and pick the most recent one
                # Convert 'Date of Transfer' to datetime, use .loc[] to avoid the warning
                rows.loc[:, 'Date of Transfer'] = pd.to_datetime(rows['Date of Transfer'], errors='coerce')  # Convert to datetime
                # Convert to datetime
                most_recent_row = rows.sort_values(by='Date of Transfer', ascending=False).iloc[0]  # Get the most recent one

                # Extract various details from the most recent row
                saon_value = most_recent_row.get('SAON (Secondary Addressable Object Name)', "")
                paon_value = most_recent_row.get('PAON (Primary Addressable Object Name)', "")
                street_value = most_recent_row.get('Street', "")
                postcode_value = most_recent_row.get('Postcode', "")
                duration_value = most_recent_row.get('Duration', "")
                old_or_new_value = most_recent_row.get('Old/New', "")
                ppd_category_value = most_recent_row.get('PPD Category Type', "")
                property_type_value = most_recent_row.get('Property Type', "")
                date_of_transfer = most_recent_row.get('Date of Transfer', "")

                saon = str(saon_value).strip() if pd.notna(saon_value) else ""
                paon = str(paon_value).strip() if pd.notna(paon_value) else ""
                street = str(street_value).strip() if pd.notna(street_value) else ""
                postcode = str(postcode_value).strip() if pd.notna(postcode_value) else ""
                duration = str(duration_value).strip() if pd.notna(duration_value) else ""
                old_or_new = str(old_or_new_value).strip() if pd.notna(old_or_new_value) else ""
                ppd_category = str(ppd_category_value).strip() if pd.notna(ppd_category_value) else ""
                property_type = str(property_type_value).strip() if pd.notna(property_type_value) else ""
                

                if saon:
                    property_name = f"{saon}, {paon}, {street}"
                else:
                    property_name = f"{paon}, {street}"
                
                property_name = property_name.strip()

                if postcode and property_name:
                    details = scraper.scrape_property_details(postcode, property_name)
                else:
                    details = None

                if not details or pd.isna(details):
                    print(f"Couldn't get details for property name = {property_name} & postcode = {postcode} & property index = {property_index}")
                else:
                    print(f"Fetched value for property name = {property_name} & postcode = {postcode} & property index = {property_index}")
                
                estimated_value = details.get("Estimated Value", "Not Available (Not Scraped)") if details else "Not Available (Not Scraped)"
                
                # Append data including distance
                property_details_df = pd.concat([property_details_df, pd.DataFrame([{
                    'Potential_Comparable_ID': potential_comparable_id,
                    'Property_Index': property_index,
                    'Distance': distance.astype(float),  # Add distance here
                    'Postcode': postcode,
                    'Date of Transfer': date_of_transfer,
                    'PPD Category': ppd_category,
                    'Total Floor Area': details.get("Total Floor Area", "Not Available (Not Scraped)") if details else "Not Available (Not Scraped)",
                    'Habitable Rooms': details.get("Habitable Rooms", "Not Available (Not Scraped)") if details else "Not Available (Not Scraped)",
                    'Heated Rooms': details.get("Heated Rooms", "Not Available (Not Scraped)") if details else "Not Available (Not Scraped)",
                    'SAON': saon,
                    'PAON': paon,
                    'Tenure - Gov': duration,
                    'Tenure - Propcheck': details.get("Tenure", "Not Available (Not Scraped)") if details else "Not Available (Not Scraped)",
                    'New Build? - Gov': old_or_new,
                    'New Build? - Propcheck': details.get("New Build", "Not Available (Not Scraped)") if details else "Not Available (Not Scraped)",
                    'Street': street,
                    'Form': details.get("Form", "Not Available (Not Scraped)") if details else "Not Available (Not Scraped)",
                    'Type': details.get("Type", "Not Available (Not Scraped)") if details else "Not Available (Not Scraped)",
                    'Property Type - Gov': property_type,
                    'Year Built': details.get("Year Built", "Not Available (Not Scraped)") if details else " Not Available (Not Scraped)",
                    'Current Energy Rating': details.get("Current Energy Rating", "Not Available (Not Scraped)") if details else "Not Available (Not Scraped)",
                    'Estimated Value': re.sub(r"[^\d,]", "", estimated_value),
                    'Days Before MJ Valuation': (self.mj_valuation_date - date_of_transfer).days
                }])], ignore_index=True)
                
        self.potential_comparables_df=property_details_df
        return property_details_df

    def analyze_missing_data(potential_comparables_df):
        """Analyzes and prints the percentage of missing property data."""
        no_data_rows = potential_comparables_df[
            (potential_comparables_df["Total Floor Area"] == "No Data") &
            (potential_comparables_df["Habitable Rooms"] == "No Data") &
            (potential_comparables_df["Heated Rooms"] == "No Data")
        ].shape[0]

        total_rows = potential_comparables_df.shape[0]
        no_data_percentage = (no_data_rows / total_rows) * 100 if total_rows > 0 else 0

        print(f"\nPercentage of properties with 'No Data': {no_data_percentage:.2f}%")

    def get_potential_comparables(self,sample):
        subject_property_coordinates = (sample.latitude, sample.longitude)
        knn = self.initialize_knn()
        property_indices, distances = knn.knearest(subject_property_coordinates, 100, return_distances=True)
        # Convert to a dictionary to get unique indices and their minimum distances
        unique_distances = {}

        for index, distance in zip(property_indices, distances):
            if index in unique_distances:
                unique_distances[index] = min(unique_distances[index], distance)  # Store the min distance
            else:
                unique_distances[index] = distance


        self.potential_comparables = np.array(list(unique_distances.items()))
        property_index_value = self.sample["Property_Index"]

        # Filter out the row where the property_index matches property_index_value
        self.potential_comparables = self.potential_comparables[self.potential_comparables[:, 0] != property_index_value]
        return self.potential_comparables
        
    def preprocess_data(self,df,dataset_type):

        if dataset_type not in ["Potential Comparable", "Sample"]:
            raise ValueError("dataset_type must be either 'Potential Comparable' or 'Sample'")

        # Save a copy before one-hot encoding
        if dataset_type == "Potential Comparable":
            self.pc_before_ohe = df.copy()
        else:
            self.sample_before_ohe = df.copy()

        # Drop specific columns
        columns_to_drop = ["Potential_Comparable_ID", "Property_Index", "Postcode", "PAON", "SAON", "Street"]
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
        
        # Convert 'Total Floor Area' from '180m²' or '161m²' to float
        df["Total Floor Area"] = (
            df["Total Floor Area"]
            .str.replace("m²", "", regex=False)  # Remove "m²"
            .str.replace(",", "", regex=False)   # Remove commas (if any)
            .astype(float)                        # Convert to float
        )

        # Binary encoding for 'New Build?' and 'Tenure' columns
        if "New Build?" in df.columns:
            df["New Build?"] = df["New Build?"].map({"N": 0, "Y": 1})
        if "Tenure" in df.columns:
            df["Tenure"] = df["Tenure"].map({"F": 0, "L": 1})

        # One-Hot Encoding for 'Form', 'Type', 'Property Type'
        categories = {
            "Form": ['Detached', 'Enclosed Mid Terrace', 'Enclosed End Terrace', 'End Terrace', 'Flat/Maisonette', 'Mid Terrace', 'Semi Detached', 'Other'],
            "Type": ['Flat', 'House', 'Maisonette'],
            "Property Type": ['D', 'F', 'O', 'S', 'T']
        }

        for col, values in categories.items():
            if col in df.columns:
                for val in values:
                    new_col_name = f"{col}_{val}"
                    df[new_col_name] = (df[col] == val).astype(int)

        # Drop original categorical columns
        df.drop(columns=categories.keys(), inplace=True, errors='ignore')
        
        # Convert the object columns to numeric and ensure precision stays intact
        df[df.select_dtypes(include=['object']).columns] = \
            df.select_dtypes(include=['object']).apply(pd.to_numeric, errors='coerce')
        
        # Convert 'Distance' column to float (if it exists)
        if "Distance" in df.columns:
            df['Distance'] = df['Distance'].astype(np.float64)
        
        return df
    
    def scale_features(self,train_df, test_df, columns_to_scale):
        """
        Scales numerical features using MinMaxScaler.
        
        Parameters:
            train_df (pd.DataFrame): The DataFrame used to fit the scaler (e.g., selected_pc_df).
            test_df (pd.DataFrame): The DataFrame where the same transformation is applied (e.g., selected_sample_df).
            columns_to_scale (list): List of numeric columns to scale.

        Returns:
            pd.DataFrame, pd.DataFrame: Scaled versions of train_df and test_df.
        """
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()

        # Fit the scaler on the training DataFrame
        scaler.fit(train_df[columns_to_scale])

        # Apply transformation to both DataFrames
        scaled_train_df = train_df.copy()
        scaled_test_df = test_df.copy()
        
        scaled_train_df[columns_to_scale] = scaler.transform(train_df[columns_to_scale])
        scaled_test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])
        

        return scaled_train_df, scaled_test_df
    
    def get_comparables(self,selected_sample_df,scaled_selected_pc_df,k):
        X_subject = selected_sample_df
        X_comparables = scaled_selected_pc_df

        # Initialize the KNN model with Euclidean distance
        knn_euclidean = NearestNeighbors(n_neighbors=k, metric='euclidean')

        # Fit the model on the potential comparables data
        knn_euclidean.fit(X_comparables)

        # Find the nearest 10 neighbors
        distances_euclidean, indices_euclidean = knn_euclidean.kneighbors(X_subject)

        # Retrieve the nearest 10 Potential_Comparable_IDs using the indices
        nearest_properties_df = self.pc_before_ohe.iloc[indices_euclidean[0]]

        # Get the corresponding distances for the nearest properties
        nearest_distances = distances_euclidean[0]

        # Add the distances as a new column to the dataframe
        nearest_properties_df['KNN_Distance'] = nearest_distances

        # Sort the rows by the distance column, from closest to furthest
        nearest_properties_df_sorted = nearest_properties_df.sort_values(by='KNN_Distance')

        # Display the sorted rows

        return nearest_properties_df_sorted
    
    def clean_selected_pc_df(self,selected_pc_df):
        # Define the ID column for both cases
        id_column = "Potential_Comparable_ID"
        # Piece 1: Identifying rows where any column has "Not Available (Not Scraped)"
        mask_not_scraped = selected_pc_df.eq("Not Available (Not Scraped)")
        rows_to_drop_not_scraped = mask_not_scraped.any(axis=1)  # True if any column has this value
        dropped_ids_not_scraped = selected_pc_df.loc[rows_to_drop_not_scraped, id_column]

        dropped_data_not_scraped = self.potential_comparables_df[
            self.potential_comparables_df[id_column].isin(dropped_ids_not_scraped)
        ]

        # Remove the rows with "Not Available (Not Scraped)"
        selected_pc_df = selected_pc_df.loc[~rows_to_drop_not_scraped]

        # Piece 2: Identifying rows where any column has "No Data"
        mask_no_data = selected_pc_df.eq("No Data")
        rows_to_drop_no_data = mask_no_data.any(axis=1)  # True if any column has this value
        dropped_ids_no_data = selected_pc_df.loc[rows_to_drop_no_data, id_column]

        dropped_data_no_data = self.potential_comparables_df[
            self.potential_comparables_df[id_column].isin(dropped_ids_no_data)
        ]

        # Assign the dropped rows to self.dropped_data
        self.dropped_data = pd.concat([dropped_data_not_scraped, dropped_data_no_data], ignore_index=True)

        # Drop the rows with "No Data"
        selected_pc_df = selected_pc_df.loc[~rows_to_drop_no_data]

        # Reset the index for the cleaned dataframe
        selected_pc_df.reset_index(drop=True, inplace=True)

        # Display the number of rows removed
        print(f"Removed {len(dropped_data_not_scraped)} rows with 'Not Available (Not Scraped)'.")
        print(f"Removed {len(dropped_data_no_data)} rows with 'No Data'.")
        print(f"Total dropped rows: {len(self.dropped_data)}")

        return selected_pc_df
