# ffrd-notebooks
Notebooks for interacting with FFRD data

To acquire HDF files from s3 you will need to login to your account using AWS CLI (i.e., asw sso login), and then provide two access keys within a secrets.env file, located in the root folder of the project folder. Copy and paste both the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY without quotes or spaces and remove the <> symbols from the template version. After saving edits to the secrets.env file, rebuild the dev container to begin using the notebooks.
