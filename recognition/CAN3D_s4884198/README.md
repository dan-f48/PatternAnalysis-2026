# COMP 3710 Assignment 3 README
### Dan Fuller, Student number: 48841988

## Feasibility Checkoff

markdown version:

**Feasibility Check-Off: VQ-VAE**  
Daniel Fuller: 48841988 

**1\. User Need, Scope and Acceptance Criteria**   
**User Need**: Researchers and clinicians working with prostate MRI data cannot use real patient data because of privacy laws. Realistic generated scans would provide resources for medical research without compromising real people’s privacy.  
**Scope / Purpose**: 2D slices, VQ-VAE vs a VAE baseline, SSIM evaluation, monitoring, generation of prostate scans. The point is to show if a VQ-VAE can compress 2D HipMRI slices into a discrete set of codebook entries which can be used to generate new images. This can be compared in diversity, memorisation, and fidelity to a VAE using a continuous kl regularised latent.   
**Acceptance Criteria**: 

- Mean reconstruction SSIM \> 0.6 on test set.   
- SSIM specifically within the prostate area increases compared to the baseline.   
- No codebook collapse, check that most entries within the dataset are used.   
- Affordable training process, measured with job time and max memory allocated in RANGPUR.   
- Successful patient level split, measured with a pre-train search of datasets.    
- Generated samples are no closer to their nearest training slice than test slices are, measured by SSIM. This ensures the model is not just memorising.   
- Pairwise distance across generated samples is reported, and no mode collapse is visible in an uncurated grid. 

**2\. Model Choice and Course Concepts**   
**Dataset configuration**: HipMRI 2D slices (keras\_slices\_data), loaded from NiFTI, pre split into training, validation, and test sets. Images will be normalised to greyscale \[0,1\] and padded to ensure compression works.   
**Baseline**: Standard VAE trains on the HipMRI data. Compression to \[32x16\] and decoding back to \[256 x 128\]. VAE uses continuous latent values. Is able to generate new images using samples from the latent space.   
**Model**: Convolutional encoder to \[32x16\] latent grid, vector quantiser with a defined codebook of vectors, and a mirrored decoder. Training uses reconstruction loss plus codebook and commitment losses, with straight-through gradient estimation. EMA codebook updates are the planned default for stability. Discrete codebook entries. Generates new images using a second stage autoregressive prior over the code grid which is trained after the VQ-VAE.   
**Difficulty Band**: Hard   
**Course Concepts / Literature**: Reference \[12\] from the task sheet: “Neural discrete representation learning”. Applied course concepts: convolutional layers, autoencoders / bottlenecks, separation of encoding and decoding to create generation, parameter updating using backpropagation. 

**3\. Preliminary Feasibility Evidence**   
**Data Structure**: Already pre-split into test, train, and validate data sets, with segmentation masks defining the prostate area. No overlap between sets:  
![](../../images/overlap.png) 
**Baseline Smoke Test**: (generated samples are still essentially noise, but it’s a smoke test)  
![](../../images/smoketest.png)![](../../images/smoketestsamples.png)

**4\. Risks, Budget and Fallback**   
**Risks**: The risks associated with the VQ-VAE include: codebook collapse, slice level leakage inflating SSIM, SSIM reaching the threshold of 0.6 but images still being blurry, memorisation, and prior failing.   
**Computing Budget**: Roughly 8 hours on the RANGPUR COMP3710 cluster.   
**Next Planned Experiment**: Training the full VQ-VAE (with prior) and baseline on the same split and comparing SSIM, codebook perplexity and visual generations. Also, nearest neighbour checking to validate memorisation.   
**Fallback**: Raise latent resolution or codebook size first, then add an SSIM loss term. If prior is not working, use a smaller codebook and simpler autoregressive model. 

pdf version (converted into image):
![feasiblity](../../images/feasibility.PNG)

# Files

## modules.py 
Contains all model components. Written with pytorch. 

- Encoder: three stride-2 convolutions shrink an image of dimensions (1, 256, 128) to a latent grid of dimensions (D, 32, 16)
- Decoder: the reverse of the encoder. Using transposed convolutions and a sigmoid at the end so pixel values are between 0 and 1 

- BaseModel: plain autoencoder with a continuous latent. Used for a smoke test before creating the VAE
- BaseVAE: the baseline model for this assignment. The encoder outputs a mean and logvar per latent value, a sample is drawn with the reparamatisation trick, and a KL term i have denoted as beta pulls the latent towards N(0, 1). Image generation works by decoding random latents.

**MAIN MODEL COMPONENTS**
- VectorQuantizerEMA: replaces each of the 32 x 16 encoder vectors with its nearest entry in the codebook (512 learned entries of dimension 64). The codebook is updated with exponential moving avg, and unused entries are reseeded from encoder outputs to avoid codebook collapse (which is when a small amount of codebook entries are the only ones which ever get used). A commitment loss keeps encoder outputs close to their codes, and gradients pass through the snapping step with the straight-through estimator.
- VQVAE: encoder, quantiser and decoder in sequence. The encoder is deterministic and doesnt have a KL term, unlike BaseVAE. It can reconstruct images, but can't generate. 
- CodePrior: a small casual transformer over the 512 code indices of a grid in raster order. It is trained with cross entropy loss to predict each code from the ones before it. For generation, it samples a code grid one token at a time (with a temperature setting), and the VQ-VAE decoder turns it into an image.

**Usage of these models are throughout train.py and predict.py**


## dataset.py 
Loads and preprocesses HipMRI 2d slices (keras_slice_data) and returns usable pytorch data loaders
- load_slices: reads the nifti files with nibabel, drops any extra dimensions (as a safety feature I'm pretty sure this is overkill), and converts to fload32. Slices which aren't 256 x 128 are resized to fit the proper size, as this is what the encoders expect (this cost me an overnight RANGPUR run before I realised some were odd sizes). Each image is then min-max scaled to [0, 1]. Masks are binarised (>0) 
- pad_multiple: pads height and width to a multiple of 8 so the encoder's 3 stride-2 layers divide evenly. This is also overkill as everything is generally 256 x 128, but good as safety.
- DatasetSlice: a Dataset wrapper which returns an image, or an image mask pair when masks are loaded
- get_dataloaders: builds the train, validate, and test loaders (batch size 64, shuffled for training). It also takes a subset parameter for quick / cpu based tests and a with_masks parameter for masks 
- check_no_overlap: extracts patient IDs from the filenames with a regular expression and reports slice counts, ID counts and the overlap betwenn each pair of splits. This is mainly just to fulfil a success criteria I made in the checkin. 

For data splits I use the given ones in the task. check_no_overlap never found any overlap between them.

DATA AUDIT FINDING: There were 60 out of the 11,460 training slices which were 256 x 144. Validation and test slices were uniform. Slight horizontal squashes were applied to the misshapen files. 

**Usage of this file is to provide data to train.py**


## predict.py 
Showing example usage of your trained model. It should load your saved model
weights, run inference on test cases, print out any results, and provide visualisations where applicable.

## train.py 

Containing the source code for training, validating, testing, and saving your model. The model should be imported from modules.py and the data loader should be imported from dataset.py. Make sure to plot the losses and metrics during training.





# Process / Adjustments
### baseline vae:
- during the smoke test for the checkup, I used a small subset of the data, unaware that some of the file sizes were different (256x144). This resulted in my first complete attempt to result in an error. This was fixed with a function running in the dataloader which normalises the images to 256x128
- in the second attempt, (running for 80 epochs) the vae showed a final SSIM of 0.952. The issue here was that the weak KL term meant the latent space didn’t match the N(0,1) used for sampling, so sampling landed where the decoder hadn’t been trained. The beta value I supplied in the KL calculation was too small, and my vae essentially just became a regular autoencoder, so the randomly produced samples from the latent space were gibberish, just noise essentially, while the learned slices were accurately reconstructed: 
![sample](../../images/vae_sample.png)
![sample](../../images/vae_reconstructions.png)

- increasing the beta and providing a warmup period for the vae decreased the specificity. In an effort to increase the fidelity of generated images, I also tried reducing the latent dimension of the trained vae. This reduced reconstruction fidelity and reduced the amount that the images looked like pure noise, but it was still unable to produce coherent generated images. 
![sample](../../images/vae_sample_2.png)
![sample](../../images/vae_reconstructions_2.png)




### vq vae 
- The vq-vae requires the training of 2 different models, the actual vector quantised autoencoder, and the prior (used for generation)
- Initial adjustments for data loading had to be made, but after that I ran the training process for 80 epochs on the autoencoder, and 30 epochs on the prior (which itself relies upon the pre-trained autoencoder) 
- In the results for this, the autoencoder was reporting SSIM close to 0.8, and did not appear to be making significant improvements to this metric, but the prior still looked like it was making progress judging from training loss. The final training CE for the prior was 3.625.
- For this reason, I retrained the prior with 60 epochs. The final stats were:

autoencoder:
```
epoch: 80: train MSE 0.0028, validate SSIM 0.778, 3.0 seconds
test SSIM: 0.806
peak VRAM: 0.82 GB
train: (11460, 1, 256, 128), range [0.00, 1.00]
validate: (660, 1, 256, 128), range [0.00, 1.00]
test: (540, 1, 256, 128), range [0.00, 1.00]
codes: train (11460, 512), val (660, 512), grid 32x16
```

prior:
```
epoch 60: train CE 3.474, val CE 3.972, 14.2s
peak VRAM: 2.85 GB
generation time for 16 images: 2.5s
```
- judging by the difference in training and validation CE, there may be some overfitting, but I would not say the model is memorising images. 
- the end of training generation produced these samples:
![sample](../../images/vq_vae_samples.png)


## evaluation 
