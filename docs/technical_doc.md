# Overview
This project uses Deep Learning techniques to detection the presence of whale calls using public onshore seismograph data, which is publicy available via [Canadian National Seismograph Network (CNSN)](https://earthquakescanada.nrcan.gc.ca/stndon/CNSN-RNSC/index-en.php). We focus on the call detections of the endangered species of Blue Whales and Fin Whales in the Lower St-Lawrence Seaway. Taking waveform spectrogram as the input, the algorithm can simultaneously: 1) recognize the presence of whale calls; and 2) predict its time if a call is recognized. 

# Data
## Preparation
We retrieved daily seismic waveform data from February 2020 to January 2022 from Natural Resoures Canada's FTP server: `ftp://ftp.seismo.nrcan.gc.ca/`. Six land seismometers in the LSLS were selected: RISQ (Rimouski), CNQ (Côte-Nord), SNFQ (Sainte-Félicité), ICQ (Islets-Carribou), SMQ (St-Marguerite) and PMAQ (Port Menier Anticosti). All these seismometers have one vertical and two horizontal components, except for CNQ and SMQ which only have a vertical component. All seismometers have a sampling rate of 100 Hz.  Instrument response was not removed as the response curve within the frequency range of interest (10Hz to 32Hz) is almost flat (see [investigate_instrument_response.ipynb](../notebooks/investigate_instrument_response.ipynb)).

To prepare the data for model training and evaluation, we first filtered the raw waveform using a band-pass filter (10 Hz to 32 Hz). Then we constructed two distinct datasets, one tailored for fin whale call recognition and another for blue whale call recognition. Each dataset consists of $50\%$ positive and $50\%$ negative samples. For the fin whale call dataset, each positive sample is the spectrogram of a 2s waveform containing a fin whale call and each negative sample is the spectrogram of a 2s waveform containing only noise signal. The spectrogram was calculated with a window size of 0.5s and an overlapping window of 0.45s. For stations with multiple channels, we take the average of the spectrograms of different channels. The blue whale call dataset has the same configuration except its longer sample length of 16s. The sample length is determined based on the observation that, on average, a fin whale call has a duration of 1s ([Roy et al. 2018](https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/40803612.pdf)) and a blue whale call has a duration of 8s ([Mellinger and Clark, 2003](https://pubs.aip.org/asa/jasa/article-abstract/114/2/1108/547725/Blue-whale-Balaenoptera-musculus-sounds-from-the)). For each positive sample in both datasets, the whale call is randomly located within the time window of interest. In total, the fin whale call dataset has 186,359 samples and the blue whale call dataset has 51,537 samples. Both datasets are then partitoned into three splits: training ($80\%$), validation ($10\%$) and testing ($10\%$).

## Data quality
The data quality can be determined using two parameters: the whale index value $R$ and signal-noise-ratio $SNR$ (detailed definition can be found in [[Plourde and Nedimovic, 2022](https://d197for5662m48.cloudfront.net/documents/publicationstatus/118893/preprint_pdf/1fb191babdbd9d518829ce1e5282a4bd.pdf)]).  Figure 1 shows distribution of $R$ and $SNR$ values. 
<div align="center">
    <img src="figs/R_SNR_heatmap.png" width="100%">
    <div>
    Figure 1. Histograms and scatter plots of R and SNR values of two datasets. For easier visualization, the values are truncated between 2.5% and 97.5% quantiles. </div>
</div>

A higher $R$ and a higher $SNR$ indicates a higher data quality, as illustrated by examples in Figure 2.

<div align="center">
    <img src="figs/whale_call_examples.png" width="100%">
    <div>
    Figure 2. Random whale call examples at different quality level. </div>
</div>

# Methodology

The neural architecture used is the LSTM with 3-layers of recurrent networks and a hidden size of 128. 
<div align="center">
    <img src="figs/overview.png" width="65%">
    <div>
    Figure 3. An overview of the framework. The seismic data shown here is a sample of a blue whale call detected on station PMAQ and the timestamp of this call is '2021-10-02 07:13:33.02'. The waveform has been filtered with a  band pass filter [10, 32] HZ.</div>
</div>
A final linear layer is used to project the last hidden state of the LSRM into the target. The target is a joint task of a binary classification task on discriminating postive and negative examples and a regression task on predicting call time. The loss function to be minimized is:

$$L=L_{cls}+\lambda I_{y_{pred}=1} L_{reg}$$

where $L_{cls}$ is the binary cross-entropy loss, $L_{reg}$ is the $L_1$, $I_{y_{pred}=1}$ is the indicator function and $\lambda$ is the weighting factor, which is set to 0.5 in the experiment. The model is trained for 30 epochs with a batch size of 64. 

# Experiments and Results
Similar to other data-driven methods, the performance of this detection algorithm depends on the quality of annotated data. We focus on high quality detections instead in this prototypical development effort. Heuristically, fin whale calls with $R > 5$ and $SNR > 5$ are considered as high quality detections which have 11,565 samples. Since the same filtering threshold on blue whale calls reduces the number of samples to only 1,205, we consider blue whale calls with $R > 3$ and $SNR > 1$ are considered as high quality detections which have 16,092 samples. Table 1 shows model performance on datasets before/after filtering. Model trained on this filtered dataset achieves an accuracy of 0.973 on discriminating fin whale calls against noise signals and an accuracy of 0.938 on discriminating blue whale calls against noise signals. The mean-squared errors (MSEs) of call time prediction for fin whale calls and blue whale calls are $0.08s$ and $0.86s$, respectively. The performance is better for model trained on unfiltered dataset, despite that the unfiltered datasets have a much higher dataset size. 

| Dataset    | Data Quality | Sample Number | Accuracy | MSE|
| -------- | ------- |--------| -------- | ------- 
| blue whale call  | $R$ > 3 and  $SNR$ > 1   |16,092|0.938|0.86s|
| blue whale call  | No filtering   |51,537|0.829|0.81s|
| fin whale call  | $R$ > 5 and  $SNR$ > 5   |11,565|0.973|0.08s|
| fin whale call  | No filtering   |186,359|0.792|0.09s|