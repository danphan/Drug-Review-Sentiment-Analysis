import requests
import zipfile
from io import BytesIO

url ='https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip' 

#obtain contents of zip file located at url
url_contents = requests.get(url).content

#extract zip file into current directory
zipfile.ZipFile(BytesIO(url_contents)).extractall()
