# from googleapiclient.discovery import build
# from google.oauth2 import service_account
# SCOPES=['https://www.googleapis.com/auth/drive']
# SERVICE_ACCOUNT_FILE='final-375200-108b7e234cc4.json'
# PARENT_FOLDER_ID='1My3r7IyNwSIHHDqpmvh2xweDGdLYVviH'
# def authenticate():
#     creds=service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE,scopes=SCOPES)
#     return creds
# def upload(file_path):
#     creds=authenticate()
#     service=build('drive','v3',credentials=creds)
#     file_metadata={
#         'name':"Hello",
#         'parents': [PARENT_FOLDER_ID]
#     }
#     file=service.files().create(
#         body=file_metadata,
#         media=file_path
#     ).execute()

# upload('/home/sarvesh/Pictures/vlcsnap-2022-09-17-12h36m39s263.png')


# from googleapiclient.discovery import build
# from google.oauth2 import service_account
# from googleapiclient.http import MediaFileUpload

# SCOPES = ['https://www.googleapis.com/auth/drive']
# SERVICE_ACCOUNT_FILE = 'final-375200-108b7e234cc4.json'
# PARENT_FOLDER_ID = '1My3r7IyNwSIHHDqpmvh2xweDGdLYVviH'


# def authenticate():
#     creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#     return creds


# def upload(file_path):
#     creds = authenticate()
#     service = build('drive', 'v3', credentials=creds)
    
#     file_metadata = {
#         'name': "Hello",
#         'parents': [PARENT_FOLDER_ID]
#     }

#     media_body = media_body = MediaFileUpload(file_path, resumable=True)

#     file = service.files().create(
#         body=file_metadata,
#         media_body=media_body
#     ).execute()

# upload('/home/sarvesh/Pictures/vlcsnap-2022-09-17-12h36m39s263.png')



# from googleapiclient.discovery import build
# from google.oauth2 import service_account
# from googleapiclient.http import MediaFileUpload

# SCOPES = ['https://www.googleapis.com/auth/drive']
# SERVICE_ACCOUNT_FILE = 'final-375200-108b7e234cc4.json'
# PARENT_FOLDER_ID = '1My3r7IyNwSIHHDqpmvh2xweDGdLYVviH'

# def authenticate():
#     creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#     return creds

# def upload(file_path):
#     creds = authenticate()
#     service = build('drive', 'v3', credentials=creds)
    
#     file_metadata = {
#         'name': "Hello",
#         'parents': [PARENT_FOLDER_ID]
#     }

#     media_body = MediaFileUpload(file_path, resumable=True)

#     file = service.files().create(
#         body=file_metadata,
#         media_body=media_body
#     ).execute()

#     # Retrieve the file ID
#     file_id = file.get('id')

#     # Fetch file details to get the link
#     file_details = service.files().get(fileId=file_id, fields='webViewLink').execute()

#     # Retrieve and print the link
#     file_link = file_details.get('webViewLink')
#     if file_link:
#         print(f'File uploaded successfully. Link: {file_link}')
#     else:
#         print('Unable to retrieve the file link.')

# upload('/home/sarvesh/Pictures/vlcsnap-2022-09-17-12h36m39s263.png')


from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload
import qrcode
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'final-375200-108b7e234cc4.json'
PARENT_FOLDER_ID = '1My3r7IyNwSIHHDqpmvh2xweDGdLYVviH'

def qr(data):
    qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img.save("qrcode.png")
def authenticate():
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return creds

def upload(file_path):
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {
        'name': "manchester by the sea (again)",
        'parents': [PARENT_FOLDER_ID],
    }

    media = MediaFileUpload(file_path, resumable=True)

    file = service.files().create(
        body=file_metadata,
        media_body=media
    ).execute()

    # Set permissions to allow anyone with the link to view
    service.permissions().create(
        fileId=file['id'],
        body={'role': 'reader', 'type': 'anyone'}
    ).execute()

    # Retrieve and print the shareable link
    file_link = f'https://drive.google.com/file/d/{file["id"]}/view'
    print(f'File uploaded successfully. Shareable link: {file_link}')
    qr(file_link)
upload('/home/sarvesh/Pictures/vlcsnap-2022-09-17-12h36m39s263.png')
