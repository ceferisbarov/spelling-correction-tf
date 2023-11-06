# Spelling correction via Deep Ensembles for Azerbaijani language
Deep learning models are known to be an effective method for spelling correction. We started with an LSTM-based encoder-decoder architecture, as described in this [tutorial](https://keras.io/examples/nlp/lstm_seq2seq/). It worked relatively well for originally incorrect words, but had poor performance in retaining the originally correct words. For example:

![sample](images/sample.png)
  
In order to solve this problem, we tested Deep Ensemble architecture.

![Deep Ensemble Architecture](images/de.png "Deep Ensemble Architecture")

## Our Approach
The idea is to train multiple models. These models should have:
* Same architecture
* Trained on same data
* Initialized with **different** seeds
  
During inference, all of these base models make a prediction, and these predictions are passed on to a decision algorithm.

Original Deep Ensemble architecture suggests combining logits, but approach failed in our case. Our decision algorithm is an adjusted form of the **hard voting** system employed in bagging models.

## Results

<table>
<tr><th>Prediction per model </th><th>Prediction with multiple models</th></tr>
<tr><td>

| Models | Accuracy |
|----------|----------|
| Model 1 | 0.731 |
| Model 2 | 0.701 |
| Model 3 | 0.683 |
| Model 4 | 0.735 |
| Model 5 | 0.735 |
| Model 6 | 0.731 |
| Model 7 | 0.731 |
| Model 8 | 0.661 |

</td><td>

| N_of_models | Treshold | Accuracy |
|----------|----------|----------|
| 8 | 1 | 0.732 |
| 8 | 2 | 0.810 |
| 8 | 3 | 0.812 |
| 8 | 4 | 0.804 |
| 8 | 5 | 0.780 |
| 8 | 6 | 0.746 |
| 8 | 7 | 0.701 |
| 8 | 8 | 0.619 |


</td></tr> </table>

| Method | Best result |
|----------|----------|
| Base model | 0.735 |
| Delta | 0.757 |
| Entropy | 0.757 |
| Ensemble | 0.812 |


This new approach improves the performance considerably.
![test](results/Corr_Incorr_Plot_1-1.jpg)
![test](results/Corr_Incorr_Plot_7-3.jpg)
