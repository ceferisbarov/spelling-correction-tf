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
| Model 1 | 0.707 |
| Model 2 | 0.708 |
| Model 3 | 0.691 |
| Model 4 | 0.701 |
| Model 5 | 0.714 |
| Model 6 | 0.701 |
| Model 7 | 0.699 |
| Model 8 | 0.668 |

</td><td>

| N_of_models | Treshold | Accuracy |
|----------|----------|----------|
| 7 | 1 | 0.707 |
| 7 | 2 | 0.793 |
| 7 | 3 | 0.799 |
| 7 | 4 | 0.783 |
| 7 | 5 | 0.756 |
| 7 | 6 | 0.711 |
| 7 | 7 | 0.629 |
| - | - | - |

</td></tr> </table>

This new approach improves the performance considerably.
![test](results/Corr_Incorr_Plot_1-1.jpg)
![test](results/Corr_Incorr_Plot_7-3.jpg)
