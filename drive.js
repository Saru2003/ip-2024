// const fs = require('fs');
// const { google }= require('googleapis');

// const apikeys = require('./final-375200-108b7e234cc4.json');
// const SCOPE = ['https://www.googleapis.com/auth/drive'];

// // A Function that can provide access to google drive api
// async function authorize(){
//     const jwtClient = new google.auth.JWT(
//         apikeys.client_email,
//         null,
//         apikeys.private_key,
//         SCOPE
//     );

//     await jwtClient.authorize();

//     return jwtClient;
// }

// // A Function that will upload the desired file to google drive folder
// async function uploadFile(authClient){
//     return new Promise((resolve,rejected)=>{
//         const drive = google.drive({version:'v3',auth:authClient}); 

//         var fileMetaData = {
//             name:'some.png',    
//             parents:['1My3r7IyNwSIHHDqpmvh2xweDGdLYVviH'] // A folder ID to which file will get uploaded
//         }

//         drive.files.create({
//             resource:fileMetaData,
//             media:{
//                 body: fs.createReadStream('some.png'), // files that will get uploaded
//                 mimeType:'image/jpeg'
//             },
//             fields:'id'
//         },function(error,file){
//             if(error){
//                 return rejected(error)
//             }
//             resolve(file);
//         })
//     });
// }

// authorize().then(uploadFile).catch("error",console.error()); // function call


// drive upload success
const fs = require('fs');
const { google } = require('googleapis');

const apikeys = require('./final-375200-108b7e234cc4.json');
const SCOPE = ['https://www.googleapis.com/auth/drive'];

// A Function that can provide access to Google Drive API
async function authorize() {
    const jwtClient = new google.auth.JWT(
        apikeys.client_email,
        null,
        apikeys.private_key,
        SCOPE
    );

    await jwtClient.authorize();

    return jwtClient;
}

// A Function that will upload the desired file to a Google Drive folder
async function uploadFile(authClient) {
    return new Promise((resolve, reject) => {
        const drive = google.drive({ version: 'v3', auth: authClient });

        const fileMetaData = {
            name: 'some.png',
            parents: ['1My3r7IyNwSIHHDqpmvh2xweDGdLYVviH'] // A folder ID to which file will get uploaded
        };

        drive.files.create({
            resource: fileMetaData,
            media: {
                body: fs.createReadStream('some.png'), // Specify the correct file path
                mimeType: 'image/png' // Adjust mimeType if necessary
            },
            fields: 'id'
        }, (error, file) => {
            if (error) {
                return reject(error);
            }
            console.log('Uploaded file object:', file); // Log the file object
            const fileId = file.data.id;
            const fileLink = `https://drive.google.com/file/d/${fileId}/view`;
            console.log('File uploaded successfully. View link:', fileLink);
            resolve(file);
        });
    });
}

authorize().then(uploadFile).catch(console.error);
