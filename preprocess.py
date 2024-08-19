from sklearn.preprocessing import OneHotEncoder
encoded_target = pd.DataFrame(OneHotEncoder(drop='first', sparse_output=False).fit_transform(target_col.values.reshape(-1, 1)),
                              columns=OneHotEncoder(drop='first', sparse_output=False).get_feature_names_out([target]))
processed_data = pd.concat([processed_features, encoded_target.reset_index(drop=True)], axis=1)

from sklearn.preprocessing import LabelEncoder
encoded_target = pd.DataFrame(LabelEncoder().fit_transform(target_col), columns=[target])
processed_data = pd.concat([processed_features, encoded_target.reset_index(drop=True)], axis=1)

def preprocess_data(self, data):
        if self.target not in data.columns:
            raise KeyError(f"Target column '{self.target}' not found in the DataFrame")

        # Separate features and target
        features = data.drop(columns=[self.target])
        target_col = data[self.target]



        # Identify numerical and categorical features
        numerical_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()

        # Apply Label Encoding
        encoder = LabelEncoder()
        target_col = pd.DataFrame(encoder.fit_transform(target_col.values),
                                  columns=[self.target])
        # print(categorical_features)
        if len(categorical_features)>0:
          encoder = LabelEncoder()
          encoded_cats = pd.DataFrame(encoder.fit_transform(features[categorical_features]),
                                  columns=categorical_features)
        else:
            encoded_cats = pd.DataFrame()


        # Apply One-Hot Encoding to categorical features
        # encoder = OneHotEncoder(drop='first', sparse_output=False)
        # encoded_cats = pd.DataFrame(encoder.fit_transform(features[categorical_features]),
        #                             columns=encoder.get_feature_names_out(categorical_features))

        # Apply Standardization to numerical features
        scaler = StandardScaler()
        scaled_nums = pd.DataFrame(scaler.fit_transform(features[numerical_features]),
                                   columns=numerical_features)

        # Combine the processed numerical and categorical features
        processed_features = pd.concat([scaled_nums, encoded_cats], axis=1)

        # Remove multicollinearity using the correlation matrix approach
        corr_matrix = processed_features.corr().abs()
        to_drop = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.8:
                    to_drop.add(corr_matrix.columns[i])

        processed_features.drop(columns=to_drop, inplace=True)

        # Add target back
        processed_data = pd.concat([processed_features, target_col.reset_index(drop=True)], axis=1)

        return processed_data