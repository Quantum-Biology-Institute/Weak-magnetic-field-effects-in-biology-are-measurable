#target photoshop

function trimAndSaveAllOpenPngs() {
    // Loop through all open documents
    for (var i = 0; i < app.documents.length; i++) {
        var doc = app.documents[i];
        app.activeDocument = doc;
        
        // Ensure the file is a PNG
        if (doc.name.split('.').pop().toLowerCase() !== 'png') {
            alert("Skipping non-PNG file: " + doc.name);
            continue;
        }

        // Select the non-transparent pixels (this automatically creates a bounding box)
        doc.selection.selectAll();
        doc.selection.copy(true);  // Copy merged layers to the clipboard to ensure the selection exists
        doc.selection.deselect();  // Deselect to avoid issues

        // Apply Image > Trim... based on transparent pixels
        doc.trim(TrimType.TRANSPARENT); // This will crop the canvas to non-transparent pixels

        // Save the PNG file but keep the document open
        var pngOptions = new PNGSaveOptions();
        doc.saveAs(File(doc.fullName), pngOptions, true); // Save over the original file
    }
    
    alert("All open PNG images have been trimmed and saved.");
}

// Run the function
trimAndSaveAllOpenPngs();
