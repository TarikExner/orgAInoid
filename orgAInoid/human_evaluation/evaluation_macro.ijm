function greetUserAndGetEvalID() {
    Dialog.create("Welcome to the Organoid Evaluation!");
    Dialog.addMessage("This macro will guide you through the evaluation images.\n" +
    				  " \n" +
    				  "Please classify at least 5271 images.\n" +
    				  " \n" +
    				  "You are able to exit the classification at any time and resume later. If you resume, enter your evaluator ID and your existing results file will be found.\n" +
    				  "You will then be asked if you want to continue the analysis. To start from scratch, delete your HEAT .csv file.\n" +
    				  "   \n" + 
    				  "Once you selected the image folder, an image will open.\n" +
 					  "You are then able to set the brightness, contrast, zoom etc.\n" + "   \n" +
 					  "Once you are done looking at the image, you will be asked 6 questions:\n" +
 					  "     1. Does the organoid contain RPE?\n" +
 					  "         Explanation: Is RPE visible in this organoid? It does not matter how large the area is.\n" +
 					  "     2. Will the organoid develop RPE?\n" +
 					  "         Explanation: Do you think that this organoid will contain RPE in the future? If the organoid already contains RPE, answer yes.\n" +
 					  "     3. Current or future amount of RPE (0-3)\n" +
 					  "         Explanation: Quantify your answers from question 2:\n" +
 					  "                     class 0: Organoid does not and will not contain RPE\n" +
 					  "                     class 1: Organoid does or will have a small area of RPE\n" +
 					  "                     class 2: Organoid does or will have a medium area of RPE\n" +
 					  "                     class 3: Organoid does or will have a large area of RPE\n" +
					  "     4. Does the organoid contain a lens?\n" +
 					  "         Explanation: Is a lens visible in this organoid? It does not matter how large the area is.\n" +
 					  "     5. Will the organoid develop a lens?\n" +
 					  "         Explanation: Do you think that this organoid will contain a lens in the future? If the organoid already contains a lens, answer yes.\n" +
 					  "     6. Current or future size of the lens (0-3)\n" +
 					  "         Explanation: Quantify your answers from question 2:\n" +
 					  "                      class 0: Organoid does not and will not contain a lens\n" +
 					  "                      class 1: Organoid does or will have a lense with a small area\n" +
 					  "                      class 2: Organoid does or will have a lense with a medium area\n" +
 					  "                      class 3: Organoid does or will have a lense with a large area\n" +
 					  " \n");		  
 					  
    Dialog.addMessage("Please answer the questions to the best of your knowledge.\n" +
    				  "The results file will be saved in the image directory.\n" +
                      "Thank you very much for your efforts!");
    Dialog.show();
    
    Dialog.create("Welcome to the Organoid Evaluation!");
    Dialog.addMessage("________________________________________");
    Dialog.addMessage("Please enter your evaluator ID that was given to you by Cassian:");
    Dialog.addString("     ", "");
    Dialog.addMessage("________________________________________");
    Dialog.addMessage("Next, you will be asked to provide the input-directory for the images.");
    Dialog.show();

    evaluatorID = Dialog.getString();
    if (evaluatorID == "") {
        exit("No evaluator ID entered. Exiting.");
    }

    return evaluatorID;
}

// Function to create a new results file with a header
function createNewResultsFile(resultsPath) {
    header = "Evaluator_ID,FileName,ContainsRPE,WillDevelopRPE,RPESize,ContainsLens,WillDevelopLens,LensSize\n";
    File.saveString(header, resultsPath);
}

// Function to check if an image is already in the results file
function isImageAlreadyAnalyzed(resultsPath, imgName) {
    // Read the entire content of the results file as a string
    content = File.openAsString(resultsPath);
    
    // Split the content into lines (each line is a row in the CSV)
    lines = split(content, "\n");

    // Loop through each line, starting from the second line to skip the header
    for (i = 1; i < lines.length; i++) {
        line = lines[i];
        
        if (trim(line) == "") {
            continue;
        }
        
        // Split the line by commas to get the columns
        columns = split(line, ",");
        
        // Check if the second column exists (FileName column) and matches imgName
        if (columns.length > 1 && columns[1] == imgName) {
            return true;
        }
    }

    return false;
}

function shuffleImages(array) {
	// Fisher Yates algorithm for shuffling an array
    for (var i = array.length - 1; i > 0; i--) {
        var j = floor(random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
}

function analyzeImages(evaluatorID) {
    dir = getDirectory("Select a folder containing .tif images");
    
    resultsPath = dir + evaluatorID + "_organoid_classification.csv";
    
    if (File.exists(resultsPath)) {
        Dialog.create("Existing Results File Detected");
        Dialog.addMessage("A results file for Evaluator ID " + evaluatorID + " already exists.");
        Dialog.addChoice("Do you want to:", newArray("Continue analysis"), "Continue analysis");
        Dialog.show();

        // macro version 2: This if check does nothing anymore since
        // we do not allow for restart for safety reasons :)
        userChoice = Dialog.getChoice();
        if (userChoice == "Start from scratch") {
            deleted = File.delete(resultsPath);
            createNewResultsFile(resultsPath);
        }
    } else {
        // If no existing file, create a new one
        createNewResultsFile(resultsPath);
    }
    
    // Get the list of .tif files in the directory and shuffle the list
    list = getFileList(dir);
    list = shuffleImages(list);
    nFiles = 0;
    for (i = 0; i < list.length; i++) {
        if (endsWith(list[i], ".tif"))
            nFiles++;
    }
    if (nFiles == 0) {
        exit("No .tif images found in the folder.");
    }

    // Loop through each .tif image

    for (i = 0; i < list.length; i++) {
        if (endsWith(list[i], ".tif")) {
            imgName = list[i];
            
            // Check if the image is already analyzed
            if (isImageAlreadyAnalyzed(resultsPath, imgName)) {
                continue;  // Skip the image if it has already been analyzed
            }

            open(dir + imgName);
            selectImage(imgName);
            rename("Image");
            run("Brightness/Contrast...");
            
            waitForUser("Please have a look at this image. " +
            			" \n" +
            			"Adjust the Brightness/Contrast if necessary." +
            			" \n" +
            			" \n" +
            			"Then press Ok to enter the classfication dialogue!"); 	

			Dialog.create("Image Analysis for: " + imgName);
			
			Dialog.addRadioButtonGroup("Does the organoid contain RPE?", newArray("Yes", "No"), 1, 2, "No");
			Dialog.addRadioButtonGroup("Will the organoid develop RPE?", newArray("Yes", "No"), 1, 2, "No");
			Dialog.addRadioButtonGroup("Current or future amount of RPE (0-3):", newArray("0", "1", "2", "3"), 1, 4, "0");
			Dialog.addMessage("________________________________________");
			Dialog.addRadioButtonGroup("Does the organoid contain a lens?", newArray("Yes", "No"), 1, 2, "No");
			Dialog.addRadioButtonGroup("Will the organoid develop a lens?", newArray("Yes", "No"), 1, 2, "No");
			Dialog.addRadioButtonGroup("Current or future size of the lens (0-3):", newArray("0", "1", "2", "3"), 1, 4, "0");
			
            Dialog.show();
           
			containsRPE = Dialog.getRadioButton();
			willDevelopRPE = Dialog.getRadioButton();
			amountRPE = Dialog.getRadioButton();
			containsLens = Dialog.getRadioButton();
			willDevelopLens = Dialog.getRadioButton();
			lensSize = Dialog.getRadioButton();

            // Save results to the results.csv
            row = evaluatorID + "," + imgName + "," + containsRPE + "," + willDevelopRPE + "," +
                  amountRPE + "," + containsLens + "," + willDevelopLens + "," + lensSize;
            File.append(row, resultsPath);
            
            selectImage("Image");
            close();
        }
    }
    
    Dialog.create("Thank You!");
    Dialog.addMessage("The image analysis has been completed successfully.\n\n" +
    				  " \n" +
                      "Thank you for your effort!" +
                      " \n" +
                      " \n" +
                      "Please send the results file in the image directory to" +
                      " \n" +
                      " \n" +
                      "cassian.afting@cos.uni-heidelberg.de" +
                      " \n" +
                      " \n" +
                      "... and collect your cookies!");
    Dialog.show();
    print("Analysis complete. Results saved to " + resultsPath);
}

evaluatorID = greetUserAndGetEvalID();
analyzeImages(evaluatorID);
