var doc;

// Ask the user to select a folder to save the PNGs
var saveFolder = Folder.selectDialog("Select a folder to save the PNG files");

// Check if the user selected a folder
if (saveFolder == null) {
    alert("No folder selected. Script will not run.");
} else {
    for (var i = 0; i < app.documents.length; i++) {
        doc = app.documents[i];
        app.activeDocument = doc;

        // Play the custom action
        app.doAction("SelectCrop", "Default Actions");

        // Save the document as PNG
        var pngSaveOptions = new PNGSaveOptions();
        var fileName = doc.name.replace(/\.[^\.]+$/, ''); // Remove existing extension

        // Save path for each file
        var filePath;
        if (doc.saved) {
            // If the document is already saved, use its path
            filePath = doc.path + "/" + fileName + ".png";
        } else {
            // If the document has not been saved, use the selected folder
            filePath = saveFolder + "/" + fileName + ".png";
        }

        // Save as PNG
        try {
            doc.saveAs(new File(filePath), pngSaveOptions, true, Extension.LOWERCASE);
        } catch (e) {
            alert("Error saving document " + doc.name + ": " + e.message);
        }
    }
}
