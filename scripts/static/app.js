const input = document.getElementById('fileToUpload');
const resultsTable = document.getElementById('resultsTable');
const header = document.getElementById('tableHeader');
const previewImage = document.getElementById('previewImage');
const loader = document.getElementById('loader');
const table = resultsTable.parentElement;
const delay = 800;


const addRow = (key, val, root) => {
    let row = document.createElement('tr');
    let diseaseCell = document.createElement('td');
    let probabilityCell = document.createElement('td');
    diseaseCell.appendChild(document.createTextNode(key));
    probabilityCell.appendChild(document.createTextNode(val));
    row.appendChild(diseaseCell);
    row.appendChild(probabilityCell);
    row.className = 'row-animation';
    root.appendChild(row);
};

const clearLists = () => {
    while(resultsTable.firstChild) {
        resultsTable.removeChild(resultsTable.firstChild);
    }
    while(header.firstChild) {
        header.removeChild(header.firstChild);
    }
}

// This will upload the file after having read it
const upload = (file) => {
    fetch('/process', { // Your POST endpoint
        method: 'POST',
        body: file, // This is your file object
    }).then(response =>
        response.json() // if the response is a JSON object
    ).then(success => {
        addRow("Disease", "Probability", header);

        loader.className = 'loader hidden';
        table.className = 'visible highlight';

        Object.entries(success).forEach(([key, val], index) =>
            setTimeout(() => addRow(key, val, resultsTable), (index + 1) * delay));
        M.toast({html: 'Prediction successful'});
    }).catch(error => {
            M.toast({html: 'An error has occured (check console)'});
            console.log(error); // Handle the error response object
        }
    );
};

// Event handler executed when a file is selected
const onSelectFile = () => {
    table.className = 'hidden';
    loader.className = 'loader visible';

    if(input.files && input.files[0]) {
        let reader = new FileReader();
        reader.onload = (e) => previewImage.src = e.target.result;
        reader.readAsDataURL(input.files[0]);
    }

    clearLists();

    let data = new FormData();
    data.append('image', input.files[0]);
    upload(data);
};

// Add a listener on your input
// It will be triggered when a file will be selected
input.addEventListener('change', onSelectFile, false); 