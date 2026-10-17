# next-word-prediction
<h2>15/10/2026</h2>
<p> Project Name - Next-Word Predictor<br>
<u>Team Members-</u><br>
Pranav Maheshwari<br>
Tushar <br>
Anuj <br>
We had been allotted a Machine Learning project to  create a model that predicts the next word based on the words entered before it. The model is to be trained on a text dataset and explore how it learns patterns between the words. <br></p>
Research was done on the project and a workflow was created for the same and shared with the team.
<h2> 16/10/2026</h2>
<p> The workflow was studied and understood by the team and the project was divided into three parts, namely-<br>
<ul>
<li> Finding the dataset </li>
<li> Training the model</li>
<li>Evaluation and testing of model</li>
</ul>
The first part of project is allotted to Anuj, second to Pranav and third to Tushar.<br>
Tensorflow and its related libraries were installed after creating a virtual environment. A 4 layer model was built including the input layer. There are two hidden dense layers and an output dense layer. After building the model it was compiled by adding optimizer , loss and metrics. After compilation the model was tested using using temporary input_size and vocab_size so as to ensure that the model is working correctly and that the libraries are working properly.<br>
Once the test was done, training of model started. It involved writing the generic code which could be changed later so as to incorporate the dataset from the respective folders. </p>
<h2> 17/10/2026</h2>
<p> The dataset was received and pulled into the local. The data was split into training, validation and testing in the ratio 60:20:20. The changes were made in the training code so as to include the correct path names for the data. The code was executed and the model was trained successfully.</p>
