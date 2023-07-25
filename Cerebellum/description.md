# Description


__Training__

Five high-resolution T1-weighted MR images (resolution of 0.3 mm^3) with
cerebellum tissue labels based on the Schmahmann atlas [@Schmahmann:1999aa] are
publicly available through the Computational Brain Anatomy
(https://www.cobralab.ca/cerebellum-lobules) as described in [@Park:2014aa].   A
cerebellum-only symmetric template [@Avants:2010aa;@Tustison:2015vl] was created
from the extracted five cerebellums and their contralateral counterparts. For
gross localization of the cerebellum, a 1 mm isotropic symmetric template was
created from the five templates and their contralateral counterparts (10 total),
slightly reoriented to a more centralized location in the field of view. The
rigid transform between the cerebellum template and the whole head template was
generated for use in prediction, as described below [@Avants:2014aa].

Regional labels included 12 cortical labels per hemisphere in
addition to hemispherical white matter labels.  These were consolidated into
gray matter and white matter labels per subject.  The subject-wise transforms
from the template generation described previously were used to generate
probabilistic priors for subsequent gray matter, white matter, and CSF tissue
segmentation in both the cerebellum template and subject templates using the
ANTs Atropos tool [@Avants:2011uf].  These segmentation maps were used to
refine the regional cortical labels for each of the 5 subjects, i.e., CSF
and white matter regions were used to remove non-gray matter from the original
labelings.

The resulting set of tissue segmentations and set of cortical labels were used
in conjunction with the original 5 subject cerebellum images to create
additional training data.  These data were resampled to 0.5 mm^3 to accommodate
memory limitations although such resolution is more than sufficient for current
clinical acquisitions.    To augment these data, each of the 5 subjects were
both affinely warped and deformably warped to the space of the cerebellum
template which yielded a total of 10 data sets.  These 10 were then
contralaterally flipped to create a total of 20 subjects.  Additional,
on-the-fly consisted of both intensity-based and shape-based data augmentation.

_Intensity-based data augmentation_ consisted of randomly added noise based on
ITK functionality, simulated bias fields based on N4 bias field modeling
[@Tustison:2010ac], and histogram warping for mimicking well-known MRI
intensity nonlinearities [@Nyul:1999th;@Tustison:2021ab]. These augmentation
techniques are available in ANTsXNet (Python and R):

* simulated bias field
    * ``simulate_bias_field.py``
    * ``simulateBiasField.R``
* added noise
    * ``add_noise_to_image.py``
    * ``addNoiseToImage.R``
* MRI intensity nonlinearities
    * ``histogram_warp_image_intensities.py``
    * ``histogramWarpImageIntensities.R``

_Shape-based data augmentation_ used both random linear and nonlinear
deformations after whole-brain affine normalization to the template.  This
functionality is also instantiated within ANTsXNet:

* random spatial warping
    * ``randomly_transform_image_data.py``
    * ``randomlyTransformImageData.R``

A total of three networks were trained:

* whole cerebellum extraction,
* cerebellum tissue segmentation (CSF, gray matter, and white matter), and
* cortical regional parcellation (12 labels per hemisphere).

All networks employ a traditional 3-D U-net model [@Falk:2019aa] consisting of
five layers with 32, 64, 96, 128, and 256 filters, respectively. During the
early training stages, weighted categorical cross entropy was used with later
stages utilizing multilabel dice coefficient loss.  All training was performed
on a DGX (GPUs: 4X Tesla V100, system memory: 256 GB LRDIMM DDR4) using ANTsXNet
functionality with the training scripts available with the GitHub repository
associated with this work [@antsxukbb].  Differences between networks include
attention gating [@Schlemper:2019aa] for the whole and regional networks,
probabilistic priors were used for the tissue and regional networks as
additional input channels.

__Voxelwise Label Prediction__

Given the bilateral symmetry in both the training data and training protocol,
prediction for a single subject involves batches of 2---the original and the
contralaterally flipped version.  The final probabilistic outputs were generated
by averaging the predicted outputs.  Preprocessing includes N4 bias correction
[@Tustison:2010ac], brain extraction [@Tustison:2021ab], affine warping
[@Avants:2011wx;@Avants:2014aa] to the whole brain template and subsequent
regional deformable alignment to the cerebellum template. During prediction, the
U-net network is constructed internally and the requisite additional data (i.e.,
parameter weights and templates) are automatically downloaded from Figshare and
cached using TensorFlow/Keras utilities.


