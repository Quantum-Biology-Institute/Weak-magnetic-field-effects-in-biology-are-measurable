// Adobe Illustrator script to batch scale all opened files
var scaleFactor = 20; // Scale factor as a percentage (e.g., 50 means 50%)

for (var i = 0; i < app.documents.length; i++) {
    var doc = app.documents[i];
    var items = doc.pageItems;
    
    for (var j = 0; j < items.length; j++) {
        items[j].resize(scaleFactor, scaleFactor);
    }

    // Optional: Save the document
    //doc.save();
}